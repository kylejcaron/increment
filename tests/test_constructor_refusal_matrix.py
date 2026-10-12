"""Coded-refusal contracts across Analysis constructors."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, cast

import pandas as pd
import pytest

from increment import Analysis
from increment.errors import CapabilityError
from increment.frame import MetricSpec
from tests.analysis_factory import _native_source


def _source_daily_moments(analysis: Analysis) -> object:
    return _native_source(analysis).moments(analysis.metrics[0], grain="daily")


def _summary_analysis() -> Analysis:
    return Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "variant": ["treatment", "control"],
                "revenue": [25.0, 20.0],
            }
        ),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )


def _panel_analysis() -> Analysis:
    return Analysis.from_unit_panel(
        pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "variant": ["treatment", "treatment", "control", "control"],
                "day": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 1), date(2026, 1, 2)],
                "revenue": [25.0, 30.0, 20.0, 22.0],
            }
        ),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )


def _moments_analysis() -> Analysis:
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
    return Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control")


_BUILDERS: tuple[tuple[str, Callable[[], Analysis]], ...] = (
    ("from_unit_summary", _summary_analysis),
    ("from_unit_panel", _panel_analysis),
    ("from_moments", _moments_analysis),
)


@pytest.mark.parametrize(
    ("family", "method", "expected_code"),
    [
        (
            "from_unit_summary",
            lambda a: a.panel_sql(),
            "facade.analysis.operation",
        ),
        (
            "from_unit_panel",
            lambda a: a.materialize(),
            "facade.analysis.operation",
        ),
        (
            "from_moments",
            # The source's own grain operation, which no public Analysis method
            # requests; it is the operator contract under test here.
            lambda a: _source_daily_moments(a),
            "source.moments.grain",
        ),
    ],
)
def test_constructor_families_keep_operation_and_grain_refusals(
    family: str,
    method: Callable[[Analysis], object],
    expected_code: str,
) -> None:
    analysis = dict(_BUILDERS)[family]()
    try:
        with pytest.raises(CapabilityError) as raised:
            method(analysis)
        assert raised.value.code == expected_code
    finally:
        analysis.close()


@pytest.mark.parametrize("family", [name for name, _ in _BUILDERS])
def test_constructor_families_keep_distinct_cluster_refusals(family: str) -> None:
    analysis = dict(_BUILDERS)[family]()
    try:
        with pytest.raises(CapabilityError) as raised:
            _native_source(analysis).cluster_counts()
        assert raised.value.code in {
            "source.frame.cluster_grain",
            "source.frame_panel.cluster_grain",
            "source.sql.cluster_grain",
            "source.moments.cluster_grain",
        }
    finally:
        analysis.close()


def test_definitions_keeps_native_warehouse_cluster_contract(seeded_defs, seeded_con) -> None:
    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    try:
        with pytest.raises(CapabilityError) as raised:
            _native_source(analysis).cluster_counts()
        assert raised.value.code == "source.native.warehouse_cluster_grain"
    finally:
        analysis.close()


def test_trigger_refusal_uses_public_trigger_rates(seeded_defs, seeded_con) -> None:
    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    try:
        with pytest.raises(CapabilityError) as raised:
            analysis.trigger_rates()
        assert raised.value.code == "source.native.operation"
    finally:
        analysis.close()


def test_frame_constructor_capability_refusals_precede_frame_access() -> None:
    with pytest.raises(CapabilityError) as raised:
        Analysis.from_unit_summary(
            cast(Any, object()),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", window_days=7)],
        )
    assert raised.value.code == "source.frame.constructor"

    with pytest.raises(CapabilityError) as raised:
        Analysis.from_unit_panel(
            cast(Any, object()),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            cluster="store_id",
        )
    assert raised.value.code == "source.frame_panel.cluster_grain"
    assert raised.value.context["operation"] == "from_unit_panel(cluster=...)"
    assert raised.value.context["cluster"] == "store_id"


def _build_matrix_analysis(
    family: str,
    *,
    seeded_defs: str,
    seeded_con: Any,
) -> Analysis:
    if family == "from_unit_summary":
        return _summary_analysis()
    if family == "from_unit_panel":
        return _panel_analysis()

    if family == "from_moments":
        return _moments_analysis()
    if family == "from_definitions":
        return Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con)
    raise AssertionError(f"unknown constructor family: {family}")


# Each context mapping checks selected fields; None adds no context assertions.
_MATRIX: tuple[
    tuple[str, str, str, dict[str, object] | None],
    ...,
] = (
    # materialize
    *(
        (family, "materialize", "facade.analysis.operation", None)
        for family in ("from_unit_summary", "from_unit_panel", "from_moments")
    ),
    # SQL
    ("from_unit_summary", "sql", "source.frame.sql_unsupported", None),
    ("from_unit_panel", "sql", "source.frame.sql_unsupported", None),
    ("from_moments", "sql", "source.moments.sql", None),
    (
        "from_definitions",
        "sql",
        "source.native_sql_grain",
        {"grain": "daily", "offered": frozenset({"total", "daily", "asof"})},
    ),
    # cluster-count
    ("from_unit_summary", "cluster-count", "source.frame.cluster_grain", None),
    ("from_unit_panel", "cluster-count", "source.frame_panel.cluster_grain", None),
    ("from_moments", "cluster-count", "source.moments.cluster_grain", None),
    ("from_definitions", "cluster-count", "source.native.warehouse_cluster_grain", None),
    # trigger
    *(
        (family, "trigger", "facade.analysis.operation", None)
        for family in ("from_unit_summary", "from_unit_panel", "from_moments")
    ),
    ("from_definitions", "trigger", "source.native.operation", None),
    # sitewide
    *(
        (family, "sitewide", "facade.analysis.operation", {"operation": "sitewide_evidence"})
        for family in ("from_unit_summary", "from_unit_panel", "from_moments")
    ),
    ("from_definitions", "sitewide", "query.builders.site_volume_metric_type", None),
    # breakout
    (
        "from_unit_summary",
        "breakout",
        "facade.analysis.operation",
        {"operation": "breakout_sources"},
    ),
    ("from_unit_panel", "breakout", "source.frame.undeclared_breakout", None),
    ("from_moments", "breakout", "facade.analysis.operation", {"operation": "breakout_sources"}),
    ("from_definitions", "breakout", "source.native.operation", None),
    # grain
    (
        "from_unit_summary",
        "grain",
        "source.frame.grain",
        {"grain": "daily", "offered": frozenset({"total"})},
    ),
    (
        "from_unit_panel",
        "grain",
        "source.frame_panel.grain",
        {"grain": "weekly", "offered": frozenset({"total", "daily", "asof"})},
    ),
    (
        "from_moments",
        "grain",
        "source.moments.grain",
        {"grain": "daily", "offered": frozenset({"total"})},
    ),
    (
        "from_definitions",
        "grain",
        "source.native.grain",
        {"grain": "weekly", "offered": frozenset({"daily", "asof"})},
    ),
)


def _invoke_matrix_operation(analysis: Analysis, family: str, operation: str) -> object:
    source = _native_source(analysis)
    metric = analysis.metrics[0]
    if operation == "materialize":
        return analysis.materialize()
    if operation == "sql":
        return source.sql(grain="daily")
    if operation == "cluster-count":
        return source.cluster_counts()
    if operation == "trigger":
        return analysis.srm(expected={"control": 0.5, "treatment": 0.5}, population="triggered")
    if operation == "sitewide":
        return analysis.sitewide("d7_retention" if family == "from_definitions" else "revenue")
    if operation == "breakout":
        if family in ("from_unit_panel", "from_definitions"):
            return source.moments(metric, by=("country",))
        return analysis.run_breakout()
    if operation == "grain":
        grain = "weekly" if family in ("from_unit_panel", "from_definitions") else "daily"
        return source.moments(metric, grain=cast(Any, grain))
    raise AssertionError(f"unknown matrix operation: {operation}")


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
    analysis = _build_matrix_analysis(family, seeded_defs=seeded_defs, seeded_con=seeded_con)
    try:
        with pytest.raises(CapabilityError) as raised:
            _invoke_matrix_operation(analysis, family, operation)
        assert type(raised.value) is CapabilityError
        assert raised.value.code == expected_code
        for key, value in (expected_context or {}).items():
            assert raised.value.context[key] == value
    finally:
        analysis.close()
