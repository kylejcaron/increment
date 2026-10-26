"""Lazy estimator registry public interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from increment.estimators.registry import ESTIMATORS


def __getattr__(name: str) -> Any:
    if name in {
        "ESTIMATORS",
        "estimator_id",
        "load",
        "observational_estimator_id",
        "randomized_estimator_id",
        "routes",
    }:
        from increment.estimators import registry

        return getattr(registry, name)
    raise AttributeError(name)


__all__ = [
    "ESTIMATORS",
    "estimator_id",
    "load",
    "observational_estimator_id",
    "randomized_estimator_id",
    "routes",
]
