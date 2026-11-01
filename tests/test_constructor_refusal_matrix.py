"""Coded-refusal contracts across the public builders."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, cast

import pandas as pd
import pytest

from increment import from_definitions, from_frame, from_moments
from increment.errors import CapabilityError
from increment.estimators import sitewide
from increment.frame import MetricSpec
from increment.query import panel_sql
from increment.workflows import materialize, metrics, srm, trigger_rates
from tests.analysis_factory import bound_source


def _source_daily_moments(pair: tuple[Any, Any]) -> object:
    experiment, resolver = pair
    return bound_source(resolver).moments(metrics(experiment, resolver)[0], grain="daily")


def _summary_analysis() -> Any:
    return from_frame(
        pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "variant": ["treatment", "control"],
                "revenue": [25.0, 20.0],
            }
        ),
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )


def _panel_analysis() -> Any:
    return from_frame(
        pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "variant": ["treatment", "treatment", "control", "control"],
                "day": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 1), date(2026, 1, 2)],
                "revenue": [25.0, 30.0, 20.0, 22.0],
            }
        ),
        shape="panel",
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )


def _moments_analysis() -> Any:
    from increment.decision_wire import compiled_plan_to_json
    from increment.estimation.armstats import centered_row_from_raw_sums
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(name="revenue", entity="user_id", fact="revenue")
    wire = compiled_plan_to_json(compile_decision_plan(None, [metric]))
    rows = [
        {
            **centered_row_from_raw_sums(
                {
                    "experiment_id": "e1",
                    "metric": "revenue",
                    "group_id": group,
                    "n": 2,
                    "sum_y": total,
                    "sum_y2": total_sq,
                }
            ),
            "winsor_lower_percentile": None,
            "winsor_upper_percentile": None,
            "winsor_lower_bound": None,
            "winsor_upper_bound": None,
            "winsor_n": None,
            "winsor_n_lower": None,
            "winsor_n_upper": None,
            "moments_format": 11,
            "successes": None,
            "decision_plan": wire,
        }
        for group, total, total_sq in (
            ("treatment", 60.0, 1850.0),
            ("control", 35.5, 675.25),
        )
    ]
    return from_moments(rows, metrics={"revenue": "mean"}, control="control")


_BUILDERS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("from_unit_summary", _summary_analysis),
    ("from_unit_panel", _panel_analysis),
    ("from_moments", _moments_analysis),
)


@pytest.mark.parametrize(
    ("family", "method", "expected_code"),
    [
        (
            "from_unit_summary",
            lambda pair: panel_sql(*pair),
            "evidence.operation_unavailable",
        ),
        (
            "from_unit_panel",
            lambda pair: materialize(*pair),
            "evidence.operation_unavailable",
        ),
        (
            "from_moments",
            # The source's own grain operation, which no public operation
            # requests; it is the operator contract under test here.
            lambda pair: _source_daily_moments(pair),
            "evidence.grain_unavailable",
        ),
    ],
)
def test_constructor_families_keep_operation_and_grain_refusals(
    family: str,
    method: Callable[[tuple[Any, Any]], object],
    expected_code: str,
) -> None:
    pair = dict(_BUILDERS)[family]()
    try:
        with pytest.raises(CapabilityError) as raised:
            method(pair)
        assert raised.value.code == expected_code
    finally:
        pair[1].close()


@pytest.mark.parametrize("family", [name for name, _ in _BUILDERS])
def test_constructor_families_keep_distinct_cluster_refusals(family: str) -> None:
    _, resolver = dict(_BUILDERS)[family]()
    try:
        with pytest.raises(CapabilityError) as raised:
            bound_source(resolver).cluster_counts()
        assert raised.value.code == "evidence.cluster_unavailable"
        assert set(raised.value.context) == {"constructor", "requested", "offered", "route"}
    finally:
        resolver.close()


@pytest.mark.integration
def test_definitions_keeps_native_warehouse_cluster_contract(seeded_defs, seeded_con) -> None:
    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    try:
        with pytest.raises(CapabilityError) as raised:
            bound_source(resolver).cluster_counts()
        assert raised.value.code == "evidence.cluster_unavailable"
    finally:
        cast(Any, resolver).close()


@pytest.mark.integration
def test_trigger_refusal_uses_public_trigger_rates(seeded_defs, seeded_con) -> None:
    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    try:
        with pytest.raises(CapabilityError) as raised:
            trigger_rates(experiment, resolver)
        assert raised.value.code == "evidence.operation_unavailable"
    finally:
        cast(Any, resolver).close()


def test_frame_constructor_capability_refusals_precede_frame_access() -> None:
    with pytest.raises(CapabilityError) as raised:
        from_frame(
            cast(Any, object()),
            shape="summary",
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", window_days=7)],
        )
    assert raised.value.code == "source.frame.constructor"

    with pytest.raises(CapabilityError) as raised:
        from_frame(
            cast(Any, object()),
            shape="panel",
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            cluster="store_id",
        )
    assert raised.value.code == "evidence.cluster_unavailable"
    assert set(raised.value.context) == {"constructor", "requested", "offered", "route"}


def _build_matrix_analysis(
    family: str,
    *,
    seeded_defs: str,
    seeded_con: Any,
) -> Any:
    if family == "from_unit_summary":
        return _summary_analysis()
    if family == "from_unit_panel":
        return _panel_analysis()

    if family == "from_moments":
        return _moments_analysis()
    if family == "from_definitions":
        return from_definitions(seeded_defs, seeded_con, experiment_name="new_onboarding_v2")
    raise AssertionError(f"unknown constructor family: {family}")


# Each context mapping checks selected fields; None adds no context assertions.
_MATRIX: tuple[
    tuple[str, str, str, dict[str, object] | None],
    ...,
] = (
    # materialize
    *(
        (family, "materialize", "evidence.operation_unavailable", None)
        for family in ("from_unit_summary", "from_unit_panel", "from_moments")
    ),
    # SQL
    ("from_unit_summary", "sql", "evidence.operation_unavailable", None),
    ("from_unit_panel", "sql", "evidence.operation_unavailable", None),
    ("from_moments", "sql", "evidence.operation_unavailable", None),
    (
        "from_definitions",
        "sql",
        "evidence.grain_unavailable",
        {"requested": "daily", "offered": ("total",)},
    ),
    # cluster-count
    ("from_unit_summary", "cluster-count", "evidence.cluster_unavailable", None),
    ("from_unit_panel", "cluster-count", "evidence.cluster_unavailable", None),
    ("from_moments", "cluster-count", "evidence.cluster_unavailable", None),
    ("from_definitions", "cluster-count", "evidence.cluster_unavailable", None),
    # trigger
    *(
        (family, "trigger", "evidence.operation_unavailable", None)
        for family in ("from_unit_summary", "from_unit_panel", "from_moments")
    ),
    ("from_definitions", "trigger", "evidence.operation_unavailable", None),
    # sitewide
    *(
        (
            family,
            "sitewide",
            "evidence.operation_unavailable",
            {"requested": "sitewide_evidence"},
        )
        for family in ("from_unit_summary", "from_unit_panel", "from_moments")
    ),
    ("from_definitions", "sitewide", "query.builders.site_volume_metric_type", None),
    # breakout
    (
        "from_unit_summary",
        "breakout",
        "evidence.breakout_unavailable",
        {"requested": ("country",)},
    ),
    ("from_unit_panel", "breakout", "evidence.breakout_unavailable", None),
    ("from_moments", "breakout", "evidence.breakout_unavailable", {"requested": ("country",)}),
    ("from_definitions", "breakout", "evidence.breakout_unavailable", None),
    # grain
    (
        "from_unit_summary",
        "grain",
        "evidence.grain_unavailable",
        {"requested": "daily"},
    ),
    (
        "from_unit_panel",
        "grain",
        "evidence.grain_unavailable",
        {"requested": "weekly"},
    ),
    (
        "from_moments",
        "grain",
        "evidence.grain_unavailable",
        {"requested": "daily"},
    ),
    (
        "from_definitions",
        "grain",
        "evidence.grain_unavailable",
        {"requested": "weekly"},
    ),
)


def _invoke_matrix_operation(pair: tuple[Any, Any], family: str, operation: str) -> object:
    experiment, resolver = pair
    source = bound_source(resolver)
    metric = metrics(experiment, resolver)[0]
    if operation == "materialize":
        return materialize(experiment, resolver)
    if operation == "sql":
        return source.sql(grain="daily")
    if operation == "cluster-count":
        return source.cluster_counts()
    if operation == "trigger":
        return srm(
            experiment,
            resolver,
            expected={"control": 0.5, "treatment": 0.5},
            population="triggered",
        )
    if operation == "sitewide":
        return sitewide(
            experiment, resolver, "d7_retention" if family == "from_definitions" else "revenue"
        )
    if operation == "breakout":
        return source.moments(metric, by=("country",))
    if operation == "grain":
        grain = "weekly" if family in ("from_unit_panel", "from_definitions") else "daily"
        return source.moments(metric, grain=cast(Any, grain))
    raise AssertionError(f"unknown matrix operation: {operation}")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("family", "operation", "expected_code", "expected_context"),
    _MATRIX,
    ids=[f"{family}-{operation}" for family, operation, *_ in _MATRIX],
)
def test_constructor_operation_matrix(
    family: str,
    operation: str,
    expected_code: str,
    expected_context: dict[str, object] | None,
    seeded_defs: str,
    seeded_con: Any,
) -> None:
    pair = _build_matrix_analysis(family, seeded_defs=seeded_defs, seeded_con=seeded_con)
    try:
        with pytest.raises(CapabilityError) as raised:
            _invoke_matrix_operation(pair, family, operation)
        assert type(raised.value) is CapabilityError
        assert raised.value.code == expected_code
        for key, value in (expected_context or {}).items():
            assert raised.value.context[key] == value
    finally:
        pair[1].close()
