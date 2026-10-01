"""Tests for the dataframe entry point (``increment/frame.py``).

The load-bearing tests are ``test_moments_match_group_summary`` and
``test_daily_moments_match_daily_group_summary``: they assert the narwhals
moments equal the real ibis builders column for column. Both import ibis;
``increment/frame.py`` must not.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from increment.errors import InvalidRequestError
from increment.frame import from_unit_panel, from_unit_summary
from increment.semantics.design import AdjustmentSet, Observational, Randomized
from increment.semantics.models import AnalysisPlan

_ROWS = [
    # unit,  variant,     revenue, converted, pre_revenue, orders, pre_balanced
    ("u01", "control", 22.40, 1.0, 9.10, 3.0, 16.3333),
    ("u02", "control", 10.00, 0.0, 3.75, 1.0, 3.9333),
    ("u03", "control", 41.05, 1.0, 28.60, 5.0, 34.9833),
    ("u04", "control", 14.20, 1.0, 5.05, 2.0, 8.1333),
    ("u05", "control", 10.00, 0.0, 0.00, 1.0, 3.9333),
    ("u06", "control", 28.75, 1.0, 15.20, 4.0, 22.6833),
    ("u07", "treatment", 32.10, 1.0, 8.90, 4.0, 19.3000),
    ("u08", "treatment", 15.60, 1.0, 4.10, 2.0, 2.8000),
    ("u09", "treatment", 51.30, 1.0, 30.15, 7.0, 38.5000),
    ("u10", "treatment", 10.00, 0.0, 2.20, 1.0, -2.8000),
    ("u11", "treatment", 37.85, 1.0, 16.40, 5.0, 25.0500),
    ("u12", "treatment", 19.95, 1.0, 6.75, 3.0, 7.1500),
]


_COLUMNS = [
    "user_id",
    "variant",
    "revenue",
    "converted",
    "pre_revenue",
    "orders",
    "pre_balanced",
]


def _arrow_table() -> pa.Table:
    cols = list(zip(*_ROWS, strict=True))
    return pa.table(dict(zip(_COLUMNS, cols, strict=True)))


@pytest.fixture
def arrow_frame() -> pa.Table:
    return _arrow_table()


_PANEL_ROWS = [
    # unit, variant,     day,   revenue, orders
    ("u1", "control", "d1", 4.0, 2.0),
    ("u1", "control", "d2", 8.0, 3.0),
    ("u1", "control", "d3", 0.0, 0.0),
    ("u2", "control", "d1", 0.0, 0.0),
    ("u2", "control", "d2", 3.0, 1.0),
    ("u2", "control", "d3", 6.0, 2.0),
    ("u3", "control", "d1", 2.0, 1.0),
    ("u3", "control", "d2", 2.0, 1.0),
    ("u3", "control", "d3", 2.0, 1.0),
    ("u4", "control", "d1", 10.0, 4.0),
    ("u4", "control", "d2", 0.0, 0.0),
    ("u4", "control", "d3", 5.0, 2.0),
    ("u5", "treatment", "d1", 6.0, 3.0),
    ("u5", "treatment", "d2", 6.0, 3.0),
    ("u5", "treatment", "d3", 6.0, 3.0),
    ("u6", "treatment", "d1", 12.0, 5.0),
    ("u6", "treatment", "d2", 0.0, 0.0),
    ("u6", "treatment", "d3", 3.0, 1.0),
    ("u7", "treatment", "d1", 4.0, 2.0),
    ("u7", "treatment", "d2", 9.0, 4.0),
    ("u7", "treatment", "d3", 4.0, 2.0),
    ("u8", "treatment", "d1", 0.0, 0.0),
    ("u8", "treatment", "d2", 5.0, 2.0),
    ("u8", "treatment", "d3", 8.0, 3.0),
]


_PANEL_COLUMNS = ["user_id", "variant", "day", "revenue", "orders"]


def _panel_table(rows=None) -> pa.Table:
    rows = _PANEL_ROWS if rows is None else rows
    cols = list(zip(*rows, strict=True))
    return pa.table(dict(zip(_PANEL_COLUMNS, cols, strict=True)))


@pytest.fixture
def panel_frame() -> pa.Table:
    return _panel_table()


def _encouragement_design(window_days: int | None = None):
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    return Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=window_days),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )


def test_from_unit_summary_design_defaults_to_randomized(arrow_frame: pa.Table) -> None:
    src = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    assert src.context.design == Randomized(control_group="control")


def test_from_unit_panel_design_defaults_to_randomized(panel_frame: pa.Table) -> None:
    src = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    assert src.context.design == Randomized(control_group="control")


def test_from_unit_summary_design_control_mismatch_refuses(arrow_frame: pa.Table) -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            arrow_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            design=Randomized(control_group="treatment"),
        )
    assert exc_info.value.code == "frame.validation.design_control_group"
    assert exc_info.value.context["control"] == "control"
    assert exc_info.value.context["control_group"] == "treatment"


def test_from_unit_panel_design_control_mismatch_refuses(panel_frame: pa.Table) -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_panel(
            panel_frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            design=Randomized(control_group="treatment"),
        )
    assert exc_info.value.code == "frame.validation.design_control_group"


def test_from_unit_summary_plan_declared_false_by_default(arrow_frame: pa.Table) -> None:
    src = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    assert src.context.plan.declared is False


def test_from_unit_panel_plan_declared_false_by_default(panel_frame: pa.Table) -> None:
    src = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    assert src.context.plan.declared is False


def test_from_unit_summary_plan_resolves_roles_onto_metrics(arrow_frame: pa.Table) -> None:
    src = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean", "converted": "conversion"},
        plan=AnalysisPlan(primary="revenue"),
    )
    assert src.context.plan.declared is True
    assert src.context.plan.procedures["revenue"].role == "primary"
    assert src.context.plan.procedures["converted"].role == "secondary"


def test_from_unit_panel_plan_resolves_roles_onto_metrics(panel_frame: pa.Table) -> None:
    src = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean", "orders": "mean"},
        plan=AnalysisPlan(primary="revenue"),
    )
    assert src.context.plan.declared is True
    assert src.context.plan.procedures["revenue"].role == "primary"
    assert src.context.plan.procedures["orders"].role == "secondary"


def test_from_unit_summary_observational_design_round_trips(arrow_frame: pa.Table) -> None:
    design = Observational(
        control_group="control", adjustment=AdjustmentSet(covariates=("pre_revenue",))
    )
    src = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=design,
    )
    assert src.context.design is design


def test_from_unit_panel_encouragement_design_round_trips() -> None:
    design = _encouragement_design(window_days=3)
    rows = [
        ("u1", "treatment", 1, 5.0, 0.0),
        ("u2", "control", 1, 3.0, 0.0),
    ]
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "revenue", "clicked"],
                zip(*rows, strict=True),
                strict=True,
            )
        )
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        design=design,
    )
    assert src.context.design is design
