"""Availability checks for optional warehouse support."""

from importlib.util import find_spec

from increment.errors import CodedError, RefusalSpec, refuse

_WAREHOUSE_EXTRA_MISSING = RefusalSpec(
    "install.warehouse_extra_missing",
    CodedError,
    template="{feature} requires the warehouse extra. Install it with {install}.",
)


def require_warehouse(feature: str) -> None:
    """Refuse warehouse use without importing its optional dependencies."""
    if any(find_spec(module) is None for module in ("ibis", "sqlglot")):
        refuse(
            _WAREHOUSE_EXTRA_MISSING,
            feature=feature,
            install='pip install "increment[warehouse]"',
        )
