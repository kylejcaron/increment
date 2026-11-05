"""Availability checks for optional extras."""

from importlib.util import find_spec

from increment.errors import MissingExtraError, RefusalSpec, refuse

_WAREHOUSE_EXTRA_MISSING = RefusalSpec(
    "install.warehouse_extra_missing",
    MissingExtraError,
    template="{feature} requires the warehouse extra. Install it with {install}.",
)
_DASHBOARD_EXTRA_MISSING = RefusalSpec(
    "install.dashboard_extra_missing",
    MissingExtraError,
    template="{feature} requires the dashboard extra. Install it with {install}.",
)
_TABLES_EXTRA_MISSING = RefusalSpec(
    "install.tables_extra_missing",
    MissingExtraError,
    template="{feature} requires the tables extra. Install it with {install}.",
)


def _require(spec: RefusalSpec, modules: tuple[str, ...], feature: str, install: str) -> None:
    """Refuse when any module of an extra cannot be found; nothing is imported."""
    if any(find_spec(module) is None for module in modules):
        refuse(spec, feature=feature, install=install)


def require_warehouse(feature: str) -> None:
    """Refuse warehouse use without importing its optional dependencies."""
    _require(
        _WAREHOUSE_EXTRA_MISSING,
        ("ibis", "sqlglot"),
        feature,
        'pip install "increment[warehouse]"',
    )


def require_dashboard() -> None:
    """Refuse the dashboard package without importing its optional dependencies."""
    _require(
        _DASHBOARD_EXTRA_MISSING,
        ("marimo", "anywidget", "traitlets", "coeftable", "pandas"),
        "increment.dashboard",
        'pip install "increment[dashboard]"',
    )


def require_tables(feature: str) -> None:
    """Refuse a table renderer without importing its optional dependencies."""
    _require(
        _TABLES_EXTRA_MISSING,
        ("coeftable", "pandas"),
        feature,
        'pip install "increment[tables]"',
    )
