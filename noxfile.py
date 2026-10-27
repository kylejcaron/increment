"""Test, lint, and typecheck sessions."""

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.test_slots import WORKERS_PLACEHOLDER  # noqa: E402


def _xdist_args(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Return the standard worker arguments for the current execution lane."""
    workers = "auto" if "CI" in environ else environ.get("WORKERS", WORKERS_PLACEHOLDER)
    return ("-n", workers, "--dist", "loadgroup")


PYTEST_ARGS = _xdist_args(os.environ)


def _run_heavy(session: nox.Session, *command: str, **kwargs) -> None:
    """Acquire the machine-wide slot before running a heavy command."""
    from scripts.test_slots import _with_workers, hold_slot

    hold_slot()
    if command and command[0] == "pytest" and "--benchmark-skip" not in command:
        command = ("pytest", "--benchmark-skip", *command[1:])
    command = tuple(_with_workers(list(command)))
    session.run(*command, **kwargs)


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
    "3.12": ("marimo==0.25.0", "anywidget==0.9.0"),
    "3.13": ("marimo==0.25.0", "anywidget==0.9.0"),
    "3.14": ("marimo==0.25.0", "anywidget==0.9.0"),
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
TIER_MARKERS = {
    "fast": "not slow and not parameter_recovery and not numerical_certification and not examples and not stress and not benchmark",
    "unit": "not slow and not integration and not parameter_recovery and not numerical_certification and not examples and not stress and not benchmark",
    "integration-fast": "not slow and integration and not parameter_recovery and not numerical_certification and not examples and not stress and not benchmark",
    # The compile registry builds once in its own CI job instead of in every slow shard.
    "slow": (
        "slow and not parameter_recovery and not numerical_certification and not examples"
        " and not compile_registry and not benchmark"
    ),
    "compile-registry": "compile_registry and not benchmark",
    "parameter-recovery": "parameter_recovery and not benchmark",
    "numerical-certification": "numerical_certification and not benchmark",
    "scientific": "(parameter_recovery or numerical_certification) and not benchmark",
    "examples": "examples and not benchmark",
    "all": "(slow or not slow) and not benchmark",
    "all-except-scientific": "not parameter_recovery and not numerical_certification and not benchmark",
}


def _sync(session: nox.Session, *groups: str, extras: tuple[str, ...] = ()) -> None:
    """Sync the project's uv-managed dependency groups (and extras) into the session's venv."""
    args = ["uv", "sync", "--locked"]
    for group in groups:
        args += ["--group", group]
    for extra in extras:
        args += ["--extra", extra]
    session.run_install(*args, env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location})


def _run_pytest_tier(session: nox.Session, tier: str, *pytest_args: str) -> None:
    marker = TIER_MARKERS.get(tier)
    selection = ("-m", marker) if marker is not None else ()

    _run_heavy(session, "pytest", *selection, *pytest_args)


def _run_pytest_at_floor(
    session: nox.Session,
    tier: str,
    *pytest_args: str,
    extras: tuple[str, ...] = ("demo",),
    use_posargs: bool = True,
) -> None:
    """Sync the given extras, then run pytest against direct dependency floors.

    ``uv run --with`` overlays the floor pins on top of the synced environment
    for this invocation only, without touching the lockfile. Session arguments
    are forwarded to pytest while preserving the selected tier marker.
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
    selection = (session.posargs or pytest_args) if use_posargs else pytest_args
    if use_posargs and session.posargs and Path(session.posargs[0].split("::", 1)[0]).is_file():
        tier = "focused"
    marker = TIER_MARKERS.get(tier)
    marker_args = ("-m", marker) if marker is not None else ()
    _run_heavy(
        session,
        "uv",
        "run",
        "--no-sync",
        *with_args,
        "pytest",
        "--benchmark-skip",
        *marker_args,
        *selection,
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
    _run_pytest_tier(session, "fast", "-q", *PYTEST_ARGS)


@nox.session(python=PYTHON_VERSIONS)
def tests_full(session: nox.Session) -> None:
    """Nightly: slow + examples against locked dependencies, across every
    supported Python version. Mirrors ``make test-all`` minus both scientific
    campaigns, which run once in the scientific shard job below. Previously
    covered exact-mass and capacity cases remain in this matrix.
    """
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    # File selectors opt into a focused run; other options keep the tier marker.
    tier = "all-except-scientific"
    if session.posargs and Path(session.posargs[0].split("::", 1)[0]).is_file():
        tier = "focused"
    _run_pytest_tier(
        session,
        tier,
        *(session.posargs or ("-q", *PYTEST_ARGS)),
    )
    if tier == "focused":
        return
    # Retain the numerical cases previously covered by this matrix.
    selector = (
        "tests/estimation/test_conversion_route_calibration.py::"
        "TestEnumerationEnclosesTheExactMass::test_the_interval_holds_the_exact_rational_mass"
    )
    _run_pytest_tier(
        session,
        "numerical-certification",
        "-q",
        *PYTEST_ARGS,
        "--durations=25",
        "--junitxml=junit-numerical.xml",
        f"{selector}[central]",
        f"{selector}[rare]",
        "tests/test_parity_harness.py::test_parity_case[exact_binomial_beyond_the_former_arm_ceiling]",
    )


@nox.session(python="3.12")
def tests_numerical_certification(session: nox.Session) -> None:
    """Run the required deterministic numerical-certification tier."""
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    _run_pytest_tier(
        session,
        "numerical-certification",
        "-q",
        *PYTEST_ARGS,
        *session.posargs,
    )


@nox.session(python="3.12")
def tests_scientific(session: nox.Session) -> None:
    """Run a shard of Monte-Carlo and deterministic scientific tiers."""
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    _run_pytest_tier(
        session,
        "scientific",
        "-q",
        *PYTEST_ARGS,
        *session.posargs,
    )


@nox.session(python="3.12")
def tests_warehouse_postgres(session: nox.Session) -> None:
    """Execute real PostgreSQL probes; missing connection settings fail."""
    _sync(session, "dev", "warehouse-backends", extras=("demo",))
    _run_heavy(
        session,
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
    _run_heavy(
        session,
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
    _run_heavy(
        session,
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
    _run_heavy(
        session,
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
    _run_heavy(
        session,
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


@nox.session(python="3.12")
def tests_warehouse_compile_snowflake(session: nox.Session) -> None:
    """Compile generated statements against Snowflake without executing them."""
    _sync(session, "dev", "warehouse-backends", extras=("demo",))
    _run_heavy(
        session,
        "pytest",
        "integration/warehouse_execution/test_snowflake_compile.py",
        "-q",
        "-s",
        "-m",
        "warehouse_compile",
        "-p",
        "no:tach",
        "--durations=0",
        *session.posargs,
    )


@nox.session(python="3.12")
def tests_warehouse_compile_bigquery(session: nox.Session) -> None:
    """Compile generated statements against BigQuery without executing them."""
    _sync(session, "dev", "warehouse-backends", extras=("demo",))
    _run_heavy(
        session,
        "pytest",
        "integration/warehouse_execution/test_bigquery_compile.py",
        "-q",
        "-s",
        "-m",
        "warehouse_compile",
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
    every supported Python version, minus the scientific tiers. Previously
    covered exact-mass and capacity cases remain in this matrix.

    ``tests_floor`` above is the narrower fast-only floor session.
    This full floor session also exercises coverage checks against core,
    table and dashboard dependency floors.
    """
    _run_pytest_at_floor(
        session,
        "all-except-scientific",
        "-q",
        *PYTEST_ARGS,
        extras=("demo", "tables", "dashboard"),
    )
    if session.posargs and Path(session.posargs[0].split("::", 1)[0]).is_file():
        return
    selector = (
        "tests/estimation/test_conversion_route_calibration.py::"
        "TestEnumerationEnclosesTheExactMass::test_the_interval_holds_the_exact_rational_mass"
    )
    _run_pytest_at_floor(
        session,
        "numerical-certification",
        f"{selector}[central]",
        f"{selector}[rare]",
        "tests/test_parity_harness.py::test_parity_case[exact_binomial_beyond_the_former_arm_ceiling]",
        "-q",
        *PYTEST_ARGS,
        "--durations=25",
        "--junitxml=junit-numerical.xml",
        extras=("demo", "tables", "dashboard"),
        use_posargs=False,
    )


@nox.session(python="3.12")
def tests_parameter_recovery(session: nox.Session) -> None:
    """Run one optional shard of the Monte-Carlo tier on Python 3.12.

    Pass ``--splits N --group k`` via posargs, for example
    ``nox -s tests_parameter_recovery -- --splits 4 --group 1``. Without a
    machine-specific durations file, pytest-split distributes tests evenly.
    """
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    _run_pytest_tier(
        session,
        "parameter-recovery",
        "-q",
        *PYTEST_ARGS,
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
    _run_heavy(session, "ruff", "check", "examples")
    _run_heavy(session, "ruff", "format", "--check", "examples")
    _run_pytest_tier(session, "examples", "-q", *PYTEST_ARGS)


@nox.session(python="3.12")
def docs_snippets(session: nox.Session) -> None:
    """Execute shipped README/docs Python snippets concurrently by file."""
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    _run_pytest_tier(
        session,
        "fast",
        "README.md",
        "docs",
        "-q",
        "-n",
        "2",
        "--dist",
        "loadgroup",
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
    _run_pytest_tier(session, "fast", "tests/test_tables.py", "-q")


@nox.session(python="3.12")
def tests_slow(session: nox.Session) -> None:
    """Run functional-slow tests, excluding scientific tiers and notebooks."""
    _sync(session, "dev", extras=("demo", "tables", "dashboard"))
    _run_pytest_tier(session, "slow", "-q", *PYTEST_ARGS)


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
    "triggered_population_dashboard",
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
    _run_heavy(session, "python", "scripts/check_doc_references.py", "--check")
    session.run("python", "scripts/render_capability_table.py", "--check")
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
    _run_pytest_tier(
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
    if "--readme-example" in session.posargs:
        session.install("polars")
    with TemporaryDirectory(prefix="increment-wheel-smoke-") as tmp, session.chdir(tmp):
        session.run("python", str(script), *session.posargs)


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
