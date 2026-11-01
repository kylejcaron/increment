"""Real frame-family attempts at trigger-aware auxiliary entry points.

These constructors retain metric evidence but do not declare trigger membership.
The operations below are attempted on each public source family so consumer
metadata describes actual boundaries rather than nearby native-source tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

from increment import from_frame, from_moments, from_switchback_frame
from increment.errors import CapabilityError
from increment.estimators import sitewide
from increment.workflows import breakout_summaries, export_moments, factor_summaries
from tests.analysis_factory import bound_analysis
from tests.capability_consumers_auxiliary import assert_expected, assert_refusal
from tests.test_frame_switchback import _assignment, _rows

if TYPE_CHECKING:
    from pathlib import Path

_INGRESSES = ("summary", "panel", "moments", "switchback")
_OPERATIONS = ("breakout", "factor", "dashboard", "sitewide")


def _unit_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": ["c0", "c1", "t0", "t1"],
            "group": ["C", "C", "T", "T"],
            "revenue": [1.0, 2.0, 3.0, 4.0],
            "factor": ["a", "b", "a", "b"],
        }
    )


def _panel_rows() -> pd.DataFrame:
    units = _unit_rows()
    return pd.DataFrame(
        [
            {**row, "date": date}
            for row in units.to_dict(orient="records")
            for date in ("2025-01-01", "2025-01-02")
        ]
    )


def _analysis_for(ingress: str, tmp_path: Path) -> Any:
    if ingress == "summary":
        return from_frame(
            _unit_rows(),
            shape="summary",
            unit="unit",
            group="group",
            control="C",
            metrics={"revenue": "mean"},
        )
    if ingress == "panel":
        return from_frame(
            _panel_rows(),
            shape="panel",
            unit="unit",
            group="group",
            date="date",
            control="C",
            metrics={"revenue": "mean"},
        )
    if ingress == "moments":
        experiment, resolver = from_frame(
            _unit_rows(),
            shape="summary",
            unit="unit",
            group="group",
            control="C",
            metrics={"revenue": "mean"},
        )
        try:
            path = tmp_path / "source-moments.parquet"
            export_moments(experiment, resolver, path)
            rows = pq.read_table(path).to_pylist()
        finally:
            resolver.close()
        return from_moments(rows, control="C", metrics={"revenue": "mean"})
    if ingress == "switchback":
        from tests.test_frame_switchback import _identification

        return from_switchback_frame(
            pd.DataFrame(_rows()),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"mean": "mean"},
            identification=_identification(),
            assignment=_assignment(),
        )
    raise AssertionError(f"unexpected ingress {ingress}")


def _attempt(analysis: tuple[Any, Any], operation: str, metric: str) -> object:
    experiment, resolver = analysis
    if operation == "breakout":
        return breakout_summaries(experiment, resolver)
    if operation == "factor":
        return factor_summaries(experiment, resolver)
    if operation == "dashboard":
        return bound_analysis(resolver).dashboard_group_data(metrics=(), population="triggered")
    if operation == "sitewide":
        return sitewide(experiment, resolver, metric)
    raise AssertionError(f"unexpected operation {operation}")


@pytest.mark.parametrize("ingress", _INGRESSES)
def test_triggered_auxiliary_operations_reach_real_ingress_boundaries(
    ingress: str, tmp_path: Path
) -> None:
    """Each source is constructed once; every operation hits its actual boundary."""
    analysis = _analysis_for(ingress, tmp_path)
    try:
        for operation in _OPERATIONS:
            leaf = f"{operation}_{ingress}"
            with pytest.raises(CapabilityError) as raised:
                _attempt(analysis, operation, "mean" if ingress == "switchback" else "revenue")
            assert_refusal(
                "triggered_auxiliary_reads",
                leaf,
                raised.value,
                stage="read",
            )
    finally:
        analysis[1].close()


def _constructor_with_trigger(ingress: str, tmp_path: Path) -> Any:
    """Attempt to express trigger membership at the actual public constructor."""
    if ingress == "summary":
        return from_frame(  # ty: ignore[no-matching-overload]
            _unit_rows(),
            shape="summary",
            unit="unit",
            group="group",
            control="C",
            metrics={"revenue": "mean"},
            trigger="surface",
        )
    if ingress == "panel":
        return from_frame(  # ty: ignore[no-matching-overload]
            _panel_rows(),
            shape="panel",
            unit="unit",
            group="group",
            date="date",
            control="C",
            metrics={"revenue": "mean"},
            trigger="surface",
        )
    if ingress == "moments":
        experiment, resolver = _analysis_for("summary", tmp_path)
        try:
            path = tmp_path / "cate-source-moments.parquet"
            export_moments(experiment, resolver, path)
            rows = pq.read_table(path).to_pylist()
        finally:
            resolver.close()
        return from_moments(
            rows,
            control="C",
            metrics={"revenue": "mean"},
            trigger="surface",  # ty: ignore[unknown-argument]
        )
    if ingress == "switchback":
        from tests.test_frame_switchback import _identification

        return from_switchback_frame(
            pd.DataFrame(_rows()),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"mean": "mean"},
            identification=_identification(),
            assignment=_assignment(),
            trigger="surface",  # ty: ignore[unknown-argument]
        )
    raise AssertionError(f"unexpected ingress {ingress}")


@pytest.mark.parametrize("ingress", _INGRESSES)
def test_cate_trigger_membership_is_not_expressible_at_public_constructor(
    ingress: str, tmp_path: Path
) -> None:
    """A constructor-level trigger attempt is rejected, not misreported as CATE support."""
    leaf = f"cate_{ingress}"
    assert_expected(
        "triggered_auxiliary_reads",
        leaf,
        kind="not_comparable",
        stage="construction",
    )
    with pytest.raises(TypeError, match="unexpected keyword argument 'trigger'"):
        _constructor_with_trigger(ingress, tmp_path)
