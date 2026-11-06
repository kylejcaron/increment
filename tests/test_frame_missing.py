"""Tests for the dataframe entry point (``increment/frame.py``).

The load-bearing tests are ``test_moments_match_group_summary`` and
``test_daily_moments_match_daily_group_summary``: they assert the narwhals
moments equal the real ibis builders column for column. Both import ibis;
``increment/frame.py`` must not.
"""

from __future__ import annotations

import importlib.util
import math
import warnings
from datetime import date, timedelta
from typing import Any, Literal

import narwhals as nw
import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from increment import (
    AdjustmentSet,
    AnalysisPlan,
    IdentificationGate,
    Observational,
    View,
    from_frame,
    readouts,
    render,
    run,
)
from increment._evidence_types import MissingValueAccount
from increment.errors import (
    CapabilityError,
    CodedError,
    DefinitionError,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation.diagnostics import SRMResult
from increment.estimation.engine import Method
from increment.estimation.readout_types import render_readout_failure
from increment.frame import (
    CupedCovariate,
    FramePanelSource,
    FrameTotalsSource,
    MetricSpec,
    from_unit_panel,
    from_unit_summary,
)
from increment.readouts._run import _observed_population_label
from increment.semantics.design import Randomized
from increment.semantics.models import Metric
from tests._frame_evidence_requests import _counts, _moments, _unit_rows
from tests.capability_consumers_scalar import IMPUTE_POOLED_MEAN, refused
from tests.warning_codes import warning_codes, warning_context

_PLUGIN_TOL = 1e-6


_DESIGN = Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
)


def _sum_y(row: Any) -> float:
    """Recover sum(y) from a moments row: n*ref_y + cy1, exact and
    cancellation-free."""
    return row["n"] * row["ref_y"] + row["cy1"]


def _sum_x(row: Any) -> float:
    """Recover sum(x) from a moments row: n*ref_x + cx1."""
    return row["n"] * row["ref_x"] + row["cx1"]


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


def _metric(src: FrameTotalsSource | FramePanelSource, name: str) -> Metric:
    """Typed lookup into ``src.metrics`` - the protocol types it
    ``Sequence[object]`` (zero-dependency seam), so tests that need
    ``.name`` narrow it back explicitly rather than fighting ty at every
    call site."""
    return next(m for m in src.context.metrics if m.name == name)


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
            justification="assignment only moves revenue via uptake"
        ),
    )


def _fence_panel(observation_end: date) -> FramePanelSource:
    """u_late (treatment) exposed 2025-01-08 with window ``[0, 7)`` = dates
    01-08..01-14, revenue 2.0/day; u_early (control) exposed 01-01,
    revenue 1.0/day. Panel observed 01-01..01-14."""
    base = date(2025, 1, 1)
    rows = []
    for off in range(14):
        d0 = base + timedelta(days=off)
        rows.append(("u_late", "treatment", d0, date(2025, 1, 8), 2.0))
        rows.append(("u_early", "control", d0, base, 1.0))
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue"],
                list(zip(*rows, strict=True)),
                strict=True,
            )
        )
    )
    return from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean", window_days=7)],
        exposure_date="exposed_on",
        observation_end=observation_end,
    )


def test_censoring_fence_keeps_fully_observed_unit() -> None:
    """A unit whose ENTIRE window ``[0, 7)`` (dates 01-08..01-14) is
    observed through ``observation_end=01-14`` is kept: maturity compares
    the LAST window day against the observable bound, never the exclusive
    right edge - which dropped exactly the newest fully-observed cohort
    every day, forever."""
    src = _fence_panel(date(2025, 1, 14))
    metric = _metric(src, "revenue")
    out = {r["group_id"]: r for r in _moments(src, metric, grain="total")}
    assert out["treatment"]["n"] == 1
    assert _sum_y(out["treatment"]) == pytest.approx(14.0)  # 7 days x 2.0
    assert out["control"]["n"] == 1
    assert _sum_y(out["control"]) == pytest.approx(7.0)


def test_censoring_fence_still_drops_genuinely_immature_unit() -> None:
    """One day short (observation_end=01-13, last window day 01-14): the
    same unit's outcome is NOT final and it is dropped whole."""
    src = _fence_panel(date(2025, 1, 13))
    metric = _metric(src, "revenue")
    with pytest.warns(IncrementWarning) as rec:
        out = {r["group_id"]: r for r in _moments(src, metric, grain="total")}
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    assert "treatment" not in out or out["treatment"]["n"] == 0
    assert out["control"]["n"] == 1


def _null_group_table() -> pa.Table:
    """The phantom-arm shape: 198 control, 182 treatment, 20 null labels."""
    labels = ["control"] * 198 + ["treatment"] * 182 + [None] * 20
    return pa.table(
        {
            "user_id": [f"u{i:03d}" for i in range(400)],
            "variant": labels,
            "revenue": [float((i % 7) + 1) for i in range(400)],
        }
    )


def test_unit_counts_on_a_plain_summary_source() -> None:
    """Unit and cluster requests retain the assigned count and refusal contract."""
    from increment.evidence import CountsRequest, evidence_provider, request_evidence

    labels = ["control"] * 198 + ["treatment"] * 182
    src = from_unit_summary(
        pa.table(
            {
                "user_id": [f"u{i:03d}" for i in range(380)],
                "variant": labels,
                "revenue": [float((i % 7) + 1) for i in range(380)],
            }
        ),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    provider = evidence_provider(src)
    answer = request_evidence(provider, CountsRequest(kind="unit"))
    from increment.evidence import CountEvidence

    assert isinstance(answer, CountEvidence)
    assert answer.counts == {"control": 198, "treatment": 182}
    with pytest.raises(CapabilityError) as raised:
        request_evidence(provider, CountsRequest(kind="cluster"))
    assert raised.value.code == "evidence.cluster_unavailable"


def test_unit_counts_ignore_first_metric_skip_and_declaration_order() -> None:
    """unit_counts() is the enrolled population, not the first metric's
    observed count: a metric with missing='skip' rows the others lack
    must not shrink the sample-ratio input, and reversing the declared
    metric order must not move it."""
    orders = [float((i % 5) + 1) for i in range(200)]  # fully observed
    table = _half_null_table().append_column("orders", pa.array(orders, type=pa.float64()))
    kwargs: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "control": "control",
    }
    skip_first = [
        MetricSpec(name="revenue", missing="skip"),
        MetricSpec(name="orders", value_column="orders"),
    ]
    # Missing 0/100 control vs 50/100 treatment: chi-square 66.7 on 1 df.
    with pytest.warns(IncrementWarning) as rec_first:
        src_skip_first = from_unit_summary(table, metrics=skip_first, **kwargs)
    assert _DIFFERS_BY_ARM in warning_codes(rec_first)
    with pytest.warns(IncrementWarning) as rec_last:
        src_skip_last = from_unit_summary(table, metrics=list(reversed(skip_first)), **kwargs)
    assert _DIFFERS_BY_ARM in warning_codes(rec_last)

    enrolled = {"control": 100, "treatment": 100}
    assert _counts(src_skip_first, "unit") == enrolled
    assert _counts(src_skip_last, "unit") == enrolled
    # The skipped metric's own moments still shrink; only unit_counts is enrolled.
    revenue = _metric(src_skip_first, "revenue")
    moments = {r["group_id"]: r["n"] for r in _moments(src_skip_first, revenue)}
    assert moments == {"control": 100, "treatment": 50}


def test_null_group_label_refuses_naming_the_exclude_knob() -> None:
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            _null_group_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert error.value.code == "source.frame.unassigned"
    assert error.value.context["constructor"] == "from_unit_summary"
    assert error.value.context["group"] == "variant"


def test_null_group_label_excluded_with_accounting() -> None:
    """on_unassigned='exclude': no phantom arm, no estimate for it, no SRM
    degree of freedom - the 20 excluded units are surfaced in
    unit_counts() and on SRMResult.unassigned_units instead."""
    src = from_unit_summary(
        _null_group_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        on_unassigned="exclude",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
    )
    assert _counts(src, "unit") == {"control": 198, "treatment": 182, "(unassigned)": 20}

    estimates = readouts.run(src)
    assert [e.group_id for e in estimates] == ["treatment"]

    srm = readouts.srm(src)
    assert isinstance(srm, SRMResult)
    assert srm.df == 1  # two arms; the unassigned bucket earns no df
    assert srm.unassigned_units == 20
    assert set(srm.observed) == {"control", "treatment"}
    assert not srm.is_srm  # 198 vs 182 is unremarkable; 3-way was chi2=145.5


def test_null_group_label_panel_refuses_and_excludes() -> None:
    base = date(2025, 1, 1)
    units = [f"u{i}" for i in range(12)]
    labels: dict[str, str | None] = {
        u: ("treatment" if i % 2 else "control") for i, u in enumerate(units)
    }
    labels["u0"] = None  # every row of u0 lacks an assignment
    table = pa.table(
        {
            "user_id": [u for u in units for _ in range(2)],
            "variant": [labels[u] for u in units for _ in range(2)],
            "day": [base + timedelta(days=k) for _ in units for k in range(2)],
            "revenue": [float(k) for _ in units for k in range(2)],
        }
    )
    kwargs: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "date": "day",
        "control": "control",
        "metrics": {"revenue": "mean"},
    }
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(table, **kwargs)
    assert error.value.code == "source.frame.unassigned"
    src = from_unit_panel(table, **kwargs, on_unassigned="exclude")
    assert _counts(src, "unit") == {"control": 5, "treatment": 6, "(unassigned)": 1}


def test_null_group_label_partial_row_excludes_whole_unit_with_accounting() -> None:
    """A unit with a valid group label on one day and a null label on
    another must lose its arm and its outcome entirely - not keep the
    arm while the null-labelled row silently densifies to zero with no
    accounting. Regression: excluded_units was 0 for this shape because
    only the null-labelled ROW was dropped while the unit's other row
    kept it inside its arm."""
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "variant": ["control", None, "treatment", "treatment", "control", "control"],
            "day": [base, base + timedelta(days=1)] * 3,
            "revenue": [5.0, 9.0, 1.0, 1.0, 2.0, 2.0],
        }
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        on_unassigned="exclude",
    )
    assert _counts(src, "unit") == {"control": 1, "treatment": 1, "(unassigned)": 1}
    moments = {r["group_id"]: r for r in _moments(src, _metric(src, "revenue"))}
    assert moments["control"]["n"] == 1
    assert _sum_y(moments["control"]) == pytest.approx(4.0)  # u3 only; u1's 5.0 excluded whole


def _half_null_table() -> pa.Table:
    """The null-as-zero shape: half of treatment revenue missing."""
    revenue: list[float | None] = []
    for i in range(100):
        revenue.append(float((i % 9) + 1))  # control, fully observed
    for i in range(100):
        revenue.append(None if i < 50 else float((i % 9) + 3))
    return pa.table(
        {
            "user_id": [f"u{i:03d}" for i in range(200)],
            "variant": ["control"] * 100 + ["treatment"] * 100,
            "revenue": revenue,
        }
    )


def test_null_metric_value_refuses_naming_both_fixes() -> None:
    """The refusal names the declaration tier (missing='zero'/'skip') AND
    the helper tier (increment.estimators.impute) - every failure ships its fix."""
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            _half_null_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert exc_info.value.code == "frame.validation.metric_value_missing"
    assert exc_info.value.context["metric"] == "revenue"
    assert exc_info.value.context["column"] == "revenue"
    assert exc_info.value.context["n_missing"] == 50


def test_missing_zero_reproduces_null_as_zero_on_purpose() -> None:
    """missing='zero' is the old silent null-as-zero number as DECLARED
    semantics: lift equals the zero-fill prediction exactly, while
    missing='skip' gives the (different) observed-units answer."""
    table = _half_null_table()
    revenue = table["revenue"].to_pylist()
    t_vals = list(revenue[100:])
    c_vals = revenue[:100]
    zero_pred = (sum(v or 0.0 for v in t_vals) / 100) / (sum(c_vals) / 100) - 1
    observed_t = [v for v in t_vals if v is not None]
    cc_pred = (sum(observed_t) / len(observed_t)) / (sum(c_vals) / 100) - 1
    assert zero_pred != pytest.approx(cc_pred, rel=1e-3)  # the two semantics differ

    src_zero = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", missing="zero")],
        design=_DESIGN,
    )
    (est_zero,) = readouts.run(src_zero)
    assert est_zero.require_lift().value == pytest.approx(zero_pred, rel=_PLUGIN_TOL)

    with pytest.warns(IncrementWarning):
        src_skip = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", missing="skip")],
            design=_DESIGN,
        )
    (est_skip,) = readouts.run(src_skip)
    assert est_skip.population == _observed_population_label(
        MissingValueAccount(
            "revenue", {"control": 100, "treatment": 100}, {"control": 100, "treatment": 50}
        )
    )
    assert est_skip.require_lift().value == pytest.approx(cc_pred, rel=_PLUGIN_TOL)
    moments = {r["group_id"]: r for r in _moments(src_skip, _metric(src_skip, "revenue"))}
    assert moments["treatment"]["n"] == 50  # observed-units n, visibly reduced


def test_missing_policy_is_per_spec_not_per_column() -> None:
    """Two metrics reading ONE column under different declarations get
    different moments - the policy is applied per spec at moment
    construction, never by mutating the shared frame."""
    table = _half_null_table()
    with pytest.warns(IncrementWarning) as rec:
        src = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(name="rev_zero", value_column="revenue", missing="zero"),
                MetricSpec(name="rev_skip", value_column="revenue", missing="skip"),
            ],
        )
    assert _DIFFERS_BY_ARM in warning_codes(rec)
    zero = {r["group_id"]: r for r in _moments(src, _metric(src, "rev_zero"))}
    skip = {r["group_id"]: r for r in _moments(src, _metric(src, "rev_skip"))}
    assert zero["treatment"]["n"] == 100 and skip["treatment"]["n"] == 50
    assert _sum_y(zero["treatment"]) == pytest.approx(_sum_y(skip["treatment"]))
    zero_c = {k: v for k, v in zero["control"].items() if k != "metric"}
    skip_c = {k: v for k, v in skip["control"].items() if k != "metric"}
    assert zero_c == skip_c  # untouched arm identical, label aside


_DIFFERS_BY_ARM = "frame.validation.metric_missing_differs_by_arm"


def _score_frame(control_missing: int, treatment_missing: int) -> pl.DataFrame:
    """400 units, 200 per arm; ``score`` is null for the first units of each arm."""
    rows = []
    for arm, missing in (("control", control_missing), ("treatment", treatment_missing)):
        for i in range(200):
            rows.append(
                {
                    "user_id": f"{arm}-{i:03d}",
                    "variant": arm,
                    "revenue": float(i % 5) + (1.0 if arm == "treatment" else 0.0),
                    "score": None if i < missing else float(i % 7) + 1.0,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "variant": pl.String,
            "revenue": pl.Float64,
            "score": pl.Float64,
        },
    )


def _score_run(
    frame: pl.DataFrame,
    missing: Literal["zero", "skip"],
    *,
    score_type: Literal["mean", "quantile"] = "mean",
) -> tuple[Any, list[str], list[Any]]:
    """Build and run ``revenue`` plus ``score`` under *missing*; return the
    result, the construction warning codes and the construction warnings."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        experiment, resolver = from_frame(
            frame,
            shape="summary",
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                "revenue",
                MetricSpec(
                    name="score",
                    missing=missing,
                    type=score_type,
                    quantile=0.5 if score_type == "quantile" else None,
                ),
            ],
        )
    return run(experiment, resolver), warning_codes(recorded), list(recorded)


def _row(result: Any, metric: str, group_id: str = "treatment") -> Any:
    return next(
        row for row in result.readout() if row.metric == metric and row.group_id == group_id
    )


def _family_run(frame: pl.DataFrame) -> tuple[Any, Any, Any]:
    """Run revenue (primary) with secondaries ``score`` (skip) and ``score2``, a strong effect,
    on the whole window and on a two-segment breakout; return the result and both readouts."""
    index = pl.int_range(pl.len())
    frame = frame.with_columns(
        score2=(index % 7).cast(pl.Float64)
        + 1.0
        + pl.when(pl.col("variant") == "treatment").then(3.0).otherwise(0.0),
        segment=pl.when(index % 2 == 0).then(pl.lit("a")).otherwise(pl.lit("b")),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment, resolver = from_frame(
            frame,
            shape="summary",
            unit="user_id",
            group="variant",
            control="control",
            metrics=["revenue", MetricSpec(name="score", missing="skip"), "score2"],
            plan=AnalysisPlan(primary="revenue", secondaries=["score", "score2"], q=0.10),
            breakouts=("segment",),
        )
        whole = View.overall()
        breakout = View(view="breakout", grain="total")
        result = run(experiment.with_views(breakout), resolver)
    return result, result.readout(whole), result.readout(breakout)


def test_skip_excludes_values_only_from_its_metric() -> None:
    result, _codes, _ = _score_run(_score_frame(50, 100), "skip")
    revenue, score = _row(result, "revenue"), _row(result, "score")
    assert (revenue.n_control, revenue.n_treat) == (200, 200)
    assert revenue.population is None
    assert (score.n_control, score.n_treat) == (150, 100)
    assert score.population == _observed_population_label(
        MissingValueAccount(
            "score",
            {"control": 200, "treatment": 200},
            {"control": 150, "treatment": 100},
        )
    )

    zero, zero_codes, _ = _score_run(_score_frame(50, 100), "zero")
    assert zero_codes == []
    assert _row(zero, "score").population is None
    assert (_row(zero, "score").n_control, _row(zero, "score").n_treat) == (200, 200)


def test_skip_warns_when_missing_rates_differ_by_arm() -> None:
    # 2 x 2 table (missing, present) x (control, treatment): (50, 150), (100, 100).
    # Pearson chi-square without continuity correction is 26.67 on 1 df, p = 2.4e-7.
    _result, codes, recorded = _score_run(_score_frame(50, 100), "skip")
    assert codes.count(_DIFFERS_BY_ARM) == 1
    context = warning_context(recorded, _DIFFERS_BY_ARM)
    assert context["metric"] == "score"
    assert context["n_missing"] == {"control": 50, "treatment": 100}
    assert context["n_units"] == {"control": 200, "treatment": 200}
    assert context["p_value"] == pytest.approx(2.4e-7, rel=0.03)

    # Equal missing rates: chi-square 0, p = 1.
    _result, equal_codes, _ = _score_run(_score_frame(50, 50), "skip")
    assert _DIFFERS_BY_ARM not in equal_codes

    # 1,000 control vs 10 treatment units, one treatment value missing: the expected missing
    # count in treatment is 10/1010 < 5, so the chi-square approximation is not trusted and
    # the check is unavailable (the exact conditional probability is 10/1010 = 0.0099).
    sparse = pl.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(1010)],
            "variant": ["control"] * 1000 + ["treatment"] * 10,
            "revenue": [float(i % 5) for i in range(1010)],
            "score": [None if i == 1000 else float(i % 7) for i in range(1010)],
        },
        schema_overrides={"score": pl.Float64},
    )
    _result, sparse_codes, _ = _score_run(sparse, "skip")
    assert _DIFFERS_BY_ARM not in sparse_codes


def test_skip_missing_check_is_not_run_on_a_clustered_source() -> None:
    # 2 vs 6 of 20 units missing in each of 10 clusters per arm: the unit-level test would
    # warn, but it counts units as independent, so it is not run when a cluster is declared.
    frame = pl.DataFrame(
        [
            {
                "user_id": f"{arm}-{c}-{i}",
                "variant": arm,
                "cluster_id": f"{arm}-{c}",
                "revenue": float(i % 5) + (1.0 if arm == "treatment" else 0.0),
                "score": None if i < n_missing else float(i % 7) + 1.0,
            }
            for arm, n_missing in (("control", 2), ("treatment", 6))
            for c in range(10)
            for i in range(20)
        ],
        schema_overrides={"score": pl.Float64},
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        experiment, resolver = from_frame(
            frame,
            shape="summary",
            unit="user_id",
            group="variant",
            control="control",
            cluster="cluster_id",
            metrics=["revenue", MetricSpec(name="score", missing="skip")],
        )
        result = run(experiment, resolver)
    assert _DIFFERS_BY_ARM not in warning_codes(recorded)
    assert _row(result, "score").population == _observed_population_label(
        MissingValueAccount(
            "score", {"control": 200, "treatment": 200}, {"control": 180, "treatment": 140}
        )
    )


def test_skip_handles_all_missing_and_none_missing(tmp_path) -> None:
    import pyarrow.parquet as pq

    from increment import from_moments
    from increment.workflows import export_moments

    result, codes, _ = _score_run(_score_frame(0, 200), "skip")
    assert codes.count(_DIFFERS_BY_ARM) == 1
    treatment = _row(result, "score")
    assert treatment.lift is None
    assert treatment.failure_code == "readout.cell.missing_metric_observations"

    # Every value of an enrolled control arm missing, treatment observed: no contrast for
    # score, while revenue still estimates, on the scalar and quantile routes.
    for score_type in ("mean", "quantile"):
        result, codes, _ = _score_run(_score_frame(200, 0), "skip", score_type=score_type)
        assert codes.count(_DIFFERS_BY_ARM) == 1
        score_rows = [row for row in result.readout() if row.metric == "score"]
        assert score_rows
        assert {(row.lift, row.failure_code) for row in score_rows} == {
            (None, "readout.cell.missing_metric_observations")
        }
        assert _row(result, "revenue").lift is not None

    # The same for a percentile-winsorized score, whose cutoffs need both arms, on an
    # ordinary and a secondary readout.
    from increment.semantics.models import Winsorization

    winsorized = MetricSpec(
        name="score", missing="skip", winsorization=Winsorization(upper_percentile=0.95)
    )
    for plan in (None, AnalysisPlan(primary="revenue", secondaries=["score"])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            experiment, resolver = from_frame(
                _score_frame(200, 0),
                shape="summary",
                unit="user_id",
                group="variant",
                control="control",
                metrics=["revenue", winsorized],
                plan=plan,
            )
            rows: Any = run(experiment, resolver).readout()
        (score,) = [row for row in rows if row.metric == "score"]
        assert score.failure_code == "readout.cell.missing_metric_observations"
        assert score.failure_context["unobserved_arm"] == "control"
        assert next(row for row in rows if row.metric == "revenue").lift is not None

    # With no other metric in the run, the enrolled roster still tells the skipped control
    # values from an absent control, so the run returns the unavailable row.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment, resolver = from_frame(
            _score_frame(200, 0),
            shape="summary",
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="score", missing="skip")],
        )
        alone_rows: Any = run(experiment, resolver).readout()
    (alone,) = alone_rows
    assert alone.failure_code == "readout.cell.missing_metric_observations"

    # A control arm with no enrolled units (a non-skip metric whose control rows are absent)
    # keeps the estimator's missing-control refusal.
    experiment, resolver = from_frame(
        _score_frame(0, 0),
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics=["revenue", "score"],
    )
    export_moments(experiment, resolver, tmp_path / "moments.parquet")
    rows = [
        row
        for row in pq.read_table(tmp_path / "moments.parquet").to_pylist()
        if (row["metric"], row["group_id"]) != ("score", "control")
    ]
    replayed = from_moments(
        rows,
        metrics=[MetricSpec(name="revenue", type="mean"), MetricSpec(name="score", type="mean")],
        control="control",
    )
    with pytest.raises(CodedError) as absent_control:
        run(*replayed)
    assert absent_control.value.code == "estimation.engine.control_group_found"

    # Secondaries score (skip) and score2 (strong effect). With score's control or treatment
    # values all missing, its cell stays in the family at p = 1, so BH keeps the no-missing
    # family: whole window m = 2, score2 selected (1 * 0.10 / 2); breakout m = 6 (3 metrics x
    # 2 segments), revenue and score2 selected in each (4 * 0.10 / 6).
    for control_missing, treatment_missing in ((0, 0), (200, 0), (0, 200)):
        _result, whole, segments = _family_run(_score_frame(control_missing, treatment_missing))
        rows = {row.metric: row for row in whole if row.group_id == "treatment"}
        assert rows["revenue"].lift is not None
        assert (rows["score2"].family_size, rows["score2"].discovery) == (2, True)
        assert rows["score2"].family_threshold == pytest.approx(0.10 * 1 / 2)
        thresholds = [row.family_threshold for row in segments if row.metric == "score2"]
        assert len(thresholds) == 2
        assert all(value == pytest.approx(0.10 * 4 / 6) for value in thresholds)
        if control_missing or treatment_missing:
            assert rows["score"].failure_code == "readout.cell.missing_metric_observations"
            assert {row.excluded for row in segments if row.metric == "score"} == {
                "missing_metric_observations"
            }

    result, codes, _ = _score_run(_score_frame(200, 200), "skip")
    assert _DIFFERS_BY_ARM not in codes
    assert _row(result, "revenue").lift is not None

    _result, codes, _ = _score_run(_score_frame(0, 0), "skip")
    assert codes == []


def test_skip_failure_names_the_arm_without_observations() -> None:
    """The unavailable score cell keeps its treatment arm while its failure names the arm that
    has no observations, on an ordinary and a secondary readout, and the rendered readout says
    so."""
    ordinary, _codes, _ = _score_run(_score_frame(200, 0), "skip")
    secondary, secondary_rows, _segments = _family_run(_score_frame(200, 0))
    treatment_missing, _codes, _ = _score_run(_score_frame(0, 200), "skip")
    for result, rows, unobserved in (
        (ordinary, ordinary.readout(), "control"),
        (secondary, secondary_rows, "control"),
        (treatment_missing, treatment_missing.readout(), "treatment"),
    ):
        (row,) = [row for row in rows if row.metric == "score"]
        assert row.failure_code == "readout.cell.missing_metric_observations"
        assert (row.group_id, row.failure_context["group_id"]) == ("treatment", "treatment")
        assert row.failure_context["unobserved_arm"] == unobserved
        message = render_readout_failure(
            row.failure_code, row.failure_context, identity={"metric": row.metric}
        )
        assert message is not None
        assert message in render(result, "brief")


def test_drop_is_refused_with_route_to_skip() -> None:
    with pytest.raises(CodedError) as exc_info:
        MetricSpec(name="x", missing="drop")
    assert exc_info.value.code == "frame.metric.missing_drop_renamed"
    assert exc_info.value.context["metric"] == "x"
    assert exc_info.value.context["route"] == "missing='skip'"


def test_skip_label_reaches_frame_and_render() -> None:
    result, codes, _ = _score_run(_score_frame(50, 50), "skip")
    assert codes == []
    row = _row(result, "score")
    frame = pl.DataFrame(result.to_frame(backend="polars"))
    (population,) = frame.filter(
        (pl.col("metric") == "score") & (pl.col("group_id") == "treatment")
    )["population"].to_list()
    assert population is not None
    assert population == row.population
    (line,) = [
        line
        for line in render(result, "brief").splitlines()
        if line.startswith("score / arm=treatment / ")
    ]
    assert line.count(row.population) == 1
    assert line.count("population=") == 1


def test_skip_label_reaches_breakout_frame_and_render() -> None:
    """A skip metric's segment rows state an observed-only population (no source counts a
    segment's enrolled units); the other metrics' segment rows carry none."""
    result, _whole, _segments = _family_run(_score_frame(50, 50))
    frame = pl.DataFrame(result.to_frame(backend="polars")).filter(
        (pl.col("scope_view") == "breakout") & (pl.col("group_id") == "treatment")
    )
    populations = {
        (row["metric"], row["dimension_value"]): row["population"]
        for row in frame.iter_rows(named=True)
    }
    assert populations == {
        (metric, segment): "observed-only population; counts not recorded per segment"
        if metric == "score"
        else None
        for metric in ("revenue", "score", "score2")
        for segment in ("a", "b")
    }
    lines = render(result, "full").splitlines()
    for metric, segment in populations:
        (line,) = [
            line
            for line in lines
            if line.startswith(
                f"{metric} / arm=treatment / dimension=segment / segment={segment} / "
            )
        ]
        assert line.count("observed-only population; counts not recorded per segment") == (
            1 if metric == "score" else 0
        )


class _SaturatingWhereX:
    """Propensity 0.995 where x == 1, outside the [0.01, 0.99] overlap band; else 0.5."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.where(np.asarray(X)[:, 0] == 1.0, 0.995, 0.5)


def _trimmed_observational(
    frame: pl.DataFrame, missing: Literal["error", "skip"]
) -> tuple[Any, Any]:
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x",)),
        gate=IdentificationGate(overlap="trim"),
    )
    metric = MetricSpec(
        name="revenue",
        missing=missing,
        decision_method=Method(name="iptw", propensity_learner=_SaturatingWhereX),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment, resolver = from_frame(
            frame, shape="summary", unit="user_id", group="variant", design=design, metrics=[metric]
        )
        result: Any = run(experiment, resolver)
        (row,) = [row for row in result.readout() if row.lift is not None]
    return result, row


def test_skip_label_composes_with_overlap_trimming() -> None:
    # 20 units per arm; x == 1 (trimmed) on units 0-1 of each arm; revenue is
    # null on 2 control and 4 treatment units, none of them trimmed.
    rows = []
    for arm, missing in (("C", (5, 6)), ("T", (5, 6, 7, 8))):
        for i in range(20):
            rows.append(
                {
                    "user_id": f"{arm}{i:02d}",
                    "variant": arm,
                    "x": 1.0 if i < 2 else 0.0,
                    "revenue": None
                    if i in missing
                    else float(i % 7) + (1.5 if arm == "T" else 1.0),
                }
            )
    frame = pl.DataFrame(rows)
    result, skipped = _trimmed_observational(frame, "skip")
    _, trimmed_only = _trimmed_observational(frame.drop_nulls("revenue"), "error")
    skip_label = _observed_population_label(
        MissingValueAccount("revenue", {"C": 20, "T": 20}, {"C": 18, "T": 16})
    )
    assert trimmed_only.population is not None
    assert skipped.population == f"{skip_label}; {trimmed_only.population}"
    assert skipped.estimand == "overlap_subpopulation_ate"
    (line,) = [
        line
        for line in render(result, "brief").splitlines()
        if line.startswith("revenue / arm=T / ") and f"estimand={skipped.estimand}" in line
    ]
    assert line.count(skipped.population) == 1
    assert line.count("population=") == 1


def test_inferred_all_null_metric_obeys_missing_policy() -> None:
    """An inferred all-null metric is nullable numeric data, not an
    unsupported dtype: zero keeps every unit as zero, while skip retains no
    units and error still refuses."""
    frames: list[Any] = [
        pa.table(
            {
                "unit": [1, 2, 3, 4],
                "variant": ["control", "control", "treatment", "treatment"],
                "y": [None, None, None, None],
            }
        ),
        pa.table(
            {
                "unit": [1, 2, 3, 4],
                "variant": ["control", "control", "treatment", "treatment"],
                "y": pa.array([None, None, None, None], type=pa.float64()),
            }
        ),
        pa.table(
            {
                "unit": [1, 2, 3, 4],
                "variant": ["control", "control", "treatment", "treatment"],
                "y": [0.0, 0.0, 0.0, 0.0],
            }
        ),
    ]
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        frames.append(frames[0].to_pandas(types_mapper=pd.ArrowDtype))
    if importlib.util.find_spec("polars") is not None:
        import polars as pl

        frames.extend(
            [
                pl.DataFrame(
                    {
                        "unit": [1, 2, 3, 4],
                        "variant": ["control", "control", "treatment", "treatment"],
                        "y": [None, None, None, None],
                    }
                ),
                pl.DataFrame(
                    {
                        "unit": [1, 2, 3, 4],
                        "variant": ["control", "control", "treatment", "treatment"],
                        "y": pl.Series("y", [None, None, None, None], dtype=pl.Float64),
                    }
                ),
            ]
        )

    observed: list[list[tuple[str, int, float]]] = []
    for frame in frames:
        src = from_unit_summary(
            frame,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="zero")],
        )
        observed.append(
            sorted(
                (str(row["group_id"]), row["n"], _sum_y(row))
                for row in _moments(src, _metric(src, "y"))
            )
        )
    assert all(rows == observed[0] for rows in observed)
    assert observed[0] == [("control", 2, 0.0), ("treatment", 2, 0.0)]

    all_null = frames[0]
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        src = from_unit_summary(
            all_null,
            unit="unit",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(name="y_zero", value_column="y", missing="zero"),
                MetricSpec(name="y_skip", value_column="y", missing="skip"),
            ],
        )
    # Nothing is observed, so the missing-by-arm check has no present values to test.
    assert _DIFFERS_BY_ARM not in warning_codes(recorded)
    zero = {row["group_id"]: row for row in _moments(src, _metric(src, "y_zero"))}
    skip = list(_moments(src, _metric(src, "y_skip")))
    assert zero["treatment"]["n"] == 2
    assert skip == []
    assert _sum_y(zero["treatment"]) == pytest.approx(0.0)

    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            all_null,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="error")],
        )
    assert exc_info.value.code == "frame.validation.metric_value_missing"


def test_observed_binary_metric_remains_unsupported() -> None:
    frame = pa.table(
        {
            "unit": [1, 2, 3, 4],
            "variant": ["control", "control", "treatment", "treatment"],
            "y": [b"1", b"2", b"3", b"4"],
        }
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            frame,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="zero")],
        )
    assert exc_info.value.code == "source.frame.column_dtype"


def test_all_null_fixed_size_binary_unknown_metric_refuses_stably() -> None:
    frame = pa.table(
        {
            "unit": [1, 2, 3, 4],
            "variant": ["control", "control", "treatment", "treatment"],
            "y": pa.array([None, None, None, None], type=pa.binary(4)),
        }
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            frame,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="zero")],
        )
    assert exc_info.value.code == "source.frame.column_dtype"
    assert exc_info.value.context["column"] == "y"


def test_all_null_pandas_period_unknown_metric_refuses_stably() -> None:
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {
            "unit": [1, 2, 3, 4],
            "variant": ["control", "control", "treatment", "treatment"],
            "y": pandas.Series([pandas.NaT] * 4, dtype="period[M]"),
        }
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            frame,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="zero")],
        )
    assert exc_info.value.code == "source.frame.column_dtype"
    assert exc_info.value.context["column"] == "y"


@pytest.mark.parametrize("backend", ["arrow", "polars"])
def test_inferred_null_panel_preserves_fact_freshness(backend) -> None:
    start = date(2025, 1, 1)
    records = [
        {
            "unit": unit,
            "variant": arm,
            "day": start + timedelta(days=day),
            "exposure": start,
            "observed": 1.0 if day == 0 else None,
            "missing": None,
        }
        for unit, arm in enumerate(("control", "control", "treatment", "treatment"))
        for day in range(4)
    ]
    inferred = pa.Table.from_pylist(records)
    missing_column = inferred.schema.get_field_index("missing")
    typed = inferred.set_column(
        missing_column, "missing", pa.nulls(len(records), type=pa.float64())
    )
    zero = inferred.set_column(missing_column, "missing", pa.array([0.0] * len(records)))
    populations = []
    for frame in (inferred, typed, zero):
        native = pytest.importorskip("polars").from_arrow(frame) if backend == "polars" else frame
        source = from_unit_panel(
            native,
            unit="unit",
            group="variant",
            date="day",
            control="control",
            exposure_date="exposure",
            metrics=[
                MetricSpec(name="observed", window_days=2, missing="zero"),
                MetricSpec(name="missing", window_days=2, missing="zero"),
            ],
        )
        with pytest.warns(IncrementWarning) as warnings:
            missing_rows = _moments(source, _metric(source, "missing"))
        populations.append(sum(row["n"] for row in missing_rows))
        assert "frame.censoring.dropped_units" in warning_codes(warnings)
        with pytest.warns(IncrementWarning) as observed_warnings:
            observed_rows = _moments(source, _metric(source, "observed"))
        assert warning_codes(observed_warnings) == ["frame.censoring.dropped_units"]
        assert sum(row["n"] for row in observed_rows) == 0
    assert populations == [4, 4, 4]


def test_nan_and_null_agree_across_backends() -> None:
    """The backend-divergence repro: one NaN y gave a y sum of 7.0 on
    pandas but NaN on polars/pyarrow. Both NaN and null are now one
    'missing' concept: identical refusal text by default, identical
    moments under missing='zero', on every installed backend."""
    frames: list[tuple[str, Any]] = [
        (
            "pyarrow-nan",
            pa.table(
                {
                    "unit": ["a", "b", "c", "d"],
                    "variant": ["control", "control", "treatment", "treatment"],
                    "y": [1.0, float("nan"), 3.0, 4.0],
                }
            ),
        )
    ]
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        frames.append(
            (
                "pandas-nan",
                pd.DataFrame(
                    {
                        "unit": ["a", "b", "c", "d"],
                        "variant": ["control", "control", "treatment", "treatment"],
                        "y": [1.0, float("nan"), 3.0, 4.0],
                    }
                ),
            )
        )
    if importlib.util.find_spec("polars") is not None:
        import polars as pl

        frames.append(
            (
                "polars-nan",
                pl.DataFrame(
                    {
                        "unit": ["a", "b", "c", "d"],
                        "variant": ["control", "control", "treatment", "treatment"],
                        "y": [1.0, float("nan"), 3.0, 4.0],
                    }
                ),
            )
        )
        frames.append(
            (
                "polars-null",
                pl.DataFrame(
                    {
                        "unit": ["a", "b", "c", "d"],
                        "variant": ["control", "control", "treatment", "treatment"],
                        "y": [1.0, None, 3.0, 4.0],
                    }
                ),
            )
        )
    assert len(frames) >= 2

    codes: set[str] = set()
    moment_sets: list[list[tuple[str, int, float]]] = []
    for _, frame in frames:
        with pytest.raises(InvalidRequestError) as exc_info:
            from_unit_summary(
                frame, unit="unit", group="variant", control="control", metrics={"y": "mean"}
            )
        codes.add(exc_info.value.code)
        src = from_unit_summary(
            frame,
            unit="unit",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", missing="zero")],
        )
        moment_sets.append(
            sorted(
                (str(r["group_id"]), r["n"], _sum_y(r)) for r in _moments(src, _metric(src, "y"))
            )
        )
    assert codes == {"frame.validation.metric_value_missing"}  # identical refusal on every backend
    assert all(m == moment_sets[0] for m in moment_sets)
    assert moment_sets[0] == [("control", 2, 1.0), ("treatment", 2, 7.0)]


def test_duplicate_metric_names_refused() -> None:
    """Duplicate names cross-pair arms downstream (previously measured: 4 estimates
    [0.1683, 0.1683, 1.3365, 1.3365] from one duplicated name)."""
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            _arrow_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                {"name": "m", "value_column": "revenue"},
                {"name": "m", "value_column": "orders"},
            ],
        )
    assert exc_info.value.code == "frame.duplicate_metric_name"
    assert exc_info.value.context["name"] == "m"


def _cuped_null_covariate_frame(control_nulls: int = 50) -> pa.Table:
    """True lift 0, x strongly predictive of y, *control_nulls* covariate nulls in
    the control arm only - the shape that silently biased a true-zero lift to -7.2%."""
    rng = np.random.default_rng(7)
    n = 500
    units, variants, ys, xs = [], [], [], []
    for group in ("control", "treatment"):
        x = rng.normal(30.0, 2.0, n)
        y = 2.0 + 0.9 * x + rng.normal(0.0, 1.0, n)
        units += [f"{group}-{i}" for i in range(n)]
        variants += [group] * n
        ys += y.tolist()
        xs += x.tolist()
    x_col: list[float | None] = list(xs)
    for i in range(control_nulls):  # nulls in the CONTROL arm's covariate
        x_col[i] = None
    return pa.table({"user_id": units, "variant": variants, "y": ys, "x": x_col})


def test_cuped_covariate_nulls_auto_imputed_with_counted_warning() -> None:
    """Nulls in one arm's covariate deflated that arm's x-mean (sums skip
    nulls, n counts all rows), which theta turned into a confident bias:
    previously measured: lift -7.2%, CI [-11.3%, -2.9%] at TRUE LIFT ZERO.
    Pooled-mean imputation (deterministic, never using arm or outcome) is
    unbiased under randomization - the estimate must cover zero again,
    and the repair must be counted and surfaced, never silent."""
    with pytest.warns(IncrementWarning) as rec:
        src = from_unit_summary(
            _cuped_null_covariate_frame(),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(
                    name="y",
                    decision_method=Method.cuped(
                        covariates=CupedCovariate(name="x", kind="continuous", missing="impute")
                    ),
                )
            ],
            design=_DESIGN,
        )
    assert "frame.validation.metric_covariate_missing_imputed" in warning_codes(rec)
    with warnings.catch_warnings(record=True) as ran:
        warnings.simplefilter("always")
        (est,) = readouts.run(src, decision_method=Method(name="cuped", variance_reduction="cuped"))
    # Missingness that depends on the arm but not on the value leaves the filled
    # covariate balanced.
    assert "estimation.cuped.covariate_imbalance" not in warning_codes(ran)
    lift = est.require_lift()
    assert lift.lb is not None and lift.ub is not None
    assert lift.lb <= 0.0 <= lift.ub
    assert abs(lift.value) < 0.02


def test_covariate_lost_in_one_arm_and_zero_filled_is_flagged() -> None:
    covariate = CupedCovariate(name="x", kind="continuous", missing="zero")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        experiment, resolver = from_frame(
            _cuped_null_covariate_frame(control_nulls=500),
            shape="summary",
            unit="user_id",
            group="variant",
            control="control",
            # The unadjusted sibling is its own estimate and has no covariate to check.
            metrics=[
                MetricSpec(
                    name="y",
                    decision_method=Method.cuped(covariates=covariate),
                    sensitivity_methods=[Method(name="unadjusted")],
                )
            ],
        )
        result = run(experiment, resolver)
    assert "frame.validation.metric_covariate_missing_imputed" in warning_codes(recorded)
    metadata = result.readout().metadata
    assert metadata is not None
    flagged = [
        record.cell.method
        for record in metadata.cells
        if any(item.code == "estimation.cuped.covariate_imbalance" for item in record.advisories)
    ]
    assert flagged == ["cuped"]


def test_all_null_covariate_refuses_with_actionable_message() -> None:
    """A covariate with NO observed values cannot yield a pooled mean; the
    refusal must name the column and the remedy on both the entry
    chokepoint and the moment-construction path (never a bare
    float(None) TypeError, which would also differ by backend)."""
    import narwhals as nw

    from increment.frame import _apply_spec_missing

    table = pa.table(
        {
            "group_id": ["control", "treatment"] * 4,
            "y": [1.0, 2.0] * 4,
            "x": [None] * 8,
        }
    )
    long = nw.from_native(table, eager_only=True).with_columns(
        nw.col("y").cast(nw.Float64), nw.col("x").cast(nw.Float64)
    )
    covariate = CupedCovariate(name="x", kind="continuous", missing="impute")
    spec = MetricSpec(name="y", decision_method=Method.cuped(covariates=covariate))
    with pytest.raises(InvalidRequestError) as exc_info:
        _apply_spec_missing(long, spec, has_x=True, has_den=False, covariate=covariate)
    assert exc_info.value.code == "frame.validation.metric_covariate_impute_all_null"
    assert exc_info.value.context == {"metric": "y", "covariate": "x"}


def test_asof_unit_rows_completed_windows_unbounded_retention_refuses() -> None:
    """``_asof_unit_rows`` carries its own completed_windows_only guard,
    independent of any caller-side pre-check."""
    import narwhals as nw

    from increment._frame_moments import _asof_unit_rows
    from increment.frame import synthesise_metric

    spec = MetricSpec(name="d0_plus", type="retention", value_column="revenue", threshold_days=0)
    metric = synthesise_metric(spec)
    panel = nw.from_native(pa.table({"unit_id": ["u1"]}), eager_only=True)
    exposure = nw.from_native(
        pa.table({"unit_id": ["u1"], "__exposure__": [date(2025, 1, 1)]}), eager_only=True
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        _asof_unit_rows(
            panel,
            spec=spec,
            metric=metric,
            by=(),
            exposure=exposure,
            uptake=None,
            uptake_window_days=None,
            completed_windows_only=True,
            identity_ordinal="__missing_ordinal__",
        )
    assert exc_info.value.code == "frame.moments.asof_moments_completed"
    assert exc_info.value.context["metric"] == "d0_plus"


def test_covariate_imputation_computed_before_outcome_skip() -> None:
    """The covariate's pooled mean must be computed from the covariate
    column alone, before missing='skip' removes any outcome-null rows -
    regression: y=[1,None,3,4], x=[0,100,None,10] previously imputed 5.0
    (mean of the two x's surviving the y-skip) instead of the correct
    36.67 (mean of all three observed x's)."""
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "y": pa.array([1.0, None, 3.0, 4.0], type=pa.float64()),
            "x": pa.array([0.0, 100.0, None, 10.0], type=pa.float64()),
        }
    )
    with pytest.warns(UserWarning):
        src = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(
                    name="y",
                    missing="skip",
                    decision_method=Method.cuped(
                        covariates=CupedCovariate(name="x", kind="continuous", missing="impute")
                    ),
                )
            ],
        )
    moments = {row["group_id"]: row for row in _moments(src, _metric(src, "y"))}
    observed_x_mean = (0.0 + 100.0 + 10.0) / 3
    assert moments["control"]["n"] == 1  # the y=None unit is skipped
    assert moments["treatment"]["n"] == 2
    assert _sum_x(moments["control"]) == pytest.approx(0.0)
    assert _sum_x(moments["treatment"]) == pytest.approx(10.0 + observed_x_mean)


def test_cuped_covariate_missing_error_escape_hatch() -> None:
    src = from_unit_summary(
        _cuped_null_covariate_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="y",
                decision_method=Method.cuped(
                    covariates=CupedCovariate(name="x", kind="continuous", missing="refuse")
                ),
            )
        ],
    )
    (row,) = readouts.run(src)
    assert row.lift is None
    assert row.failure_code == "frame.validation.metric_covariate_column"
    assert row.failure_context is not None
    assert row.failure_context["n_missing"] == 50
    assert row.failure_context["covariate"] == "x"


def test_enforce_missing_policy_all_null_impute_covariate_refuses() -> None:
    """An all-null covariate fails only its method's public readout cell."""
    table = pa.table(
        {
            "user_id": ["u1", "u2"],
            "variant": ["control", "treatment"],
            "y": pa.array([1.0, 2.0], type=pa.float64()),
            "x": pa.array([None, None], type=pa.float64()),
        }
    )
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="y",
                decision_method=Method.cuped(
                    covariates=CupedCovariate(name="x", kind="continuous", missing="impute")
                ),
            )
        ],
    )
    (row,) = readouts.run(src)
    assert row.lift is None
    assert row.failure_code == "frame.validation.metric_covariate_impute_all_null"
    assert row.failure_context is not None
    assert row.failure_context["metric"] == "y"
    assert row.failure_context["covariate"] == "x"


def test_all_null_covariate_entry_points_share_canonical_refusal_code() -> None:
    """The moment reducer and public summary readout expose one hazard."""
    import narwhals as nw

    from increment.frame import _apply_spec_missing

    long = nw.from_native(
        pa.table({"group_id": ["control", "treatment"], "y": [1.0, 2.0], "x": [None, None]}),
        eager_only=True,
    ).with_columns(nw.col("y").cast(nw.Float64), nw.col("x").cast(nw.Float64))
    covariate = CupedCovariate(name="x", kind="continuous", missing="impute")
    spec = MetricSpec(name="y", decision_method=Method.cuped(covariates=covariate))
    with pytest.raises(InvalidRequestError) as via_moments:
        _apply_spec_missing(long, spec, has_x=True, has_den=False, covariate=covariate)

    src = from_unit_summary(
        pa.table(
            {
                "user_id": ["u1", "u2"],
                "variant": ["control", "treatment"],
                "y": [1.0, 2.0],
                "x": pa.array([None, None], type=pa.float64()),
            }
        ),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[spec],
    )
    (row,) = readouts.run(src)
    assert (
        via_moments.value.code
        == row.failure_code
        == "frame.validation.metric_covariate_impute_all_null"
    )


def test_cuped_covariate_missing_zero_uses_dense_zero_convention() -> None:
    """covariate_missing='zero' mirrors the definitions path's dense x=0
    imputation: sum(x) is exactly the observed sum, n still counts every
    row, and the moments carry no NaN on any backend."""
    table = _cuped_null_covariate_frame()
    with pytest.warns(IncrementWarning) as rec:
        src = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(
                    name="y",
                    decision_method=Method.cuped(
                        covariates=CupedCovariate(name="x", kind="continuous", missing="zero")
                    ),
                )
            ],
        )
    assert "frame.validation.metric_covariate_missing_imputed" in warning_codes(rec)
    moments = {r["group_id"]: r for r in _moments(src, _metric(src, "y"))}
    observed_sum = sum(v for v in table["x"].to_pylist()[:500] if v is not None)
    assert moments["control"]["n"] == 500
    assert _sum_x(moments["control"]) == pytest.approx(observed_sum)
    assert math.isfinite(moments["control"]["cx2"])


def test_panel_null_metric_value_refuses_and_zero_fills() -> None:
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": [u for u in ("a", "b", "c", "d") for _ in range(2)],
            "variant": [
                g for g in ("control", "control", "treatment", "treatment") for _ in range(2)
            ],
            "day": [base + timedelta(days=k) for _ in range(4) for k in range(2)],
            "revenue": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    kwargs: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "date": "day",
        "control": "control",
    }
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_panel(table, metrics={"revenue": "mean"}, **kwargs)
    assert exc_info.value.code == "frame.validation.metric_value_missing"
    assert exc_info.value.context["metric"] == "revenue"
    assert exc_info.value.context["column"] == "revenue"
    assert exc_info.value.context["n_missing"] == 1
    src = from_unit_panel(table, metrics=[MetricSpec(name="revenue", missing="zero")], **kwargs)
    out = {r["group_id"]: r for r in _moments(src, _metric(src, "revenue"), grain="total")}
    assert _sum_y(out["control"]) == pytest.approx(7.0)  # 1+2 + (0+4)
    assert out["control"]["n"] == 2


def test_panel_missing_skip_refused_as_indistinguishable_from_zero() -> None:
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": ["a", "b"],
            "variant": ["control", "treatment"],
            "day": [base, base],
            "revenue": [1.0, 2.0],
        }
    )
    with pytest.raises(UnsupportedRequestError) as exc:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="revenue", missing="skip")],
        )
    assert exc.value.code == "frame.missing_policy.panel_skip"
    assert exc.value.context["missing"] == "skip"


def test_unit_frame_honours_declared_missing_policy() -> None:
    """unit_frame must serve the same declared semantics as the moments:
    zero-filled values under missing='zero', observed rows only under
    missing='skip' - never a raw null the chokepoint already ruled on."""
    table = _half_null_table()
    src_zero = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", missing="zero")],
    )
    uf = nw.from_native(_unit_rows(src_zero, _metric(src_zero, "revenue")), eager_only=True)
    assert uf.shape[0] == 200
    assert float(uf["y"].sum()) == pytest.approx(
        sum(v or 0.0 for v in table["revenue"].to_pylist())
    )
    with pytest.warns(IncrementWarning) as rec:
        src_skip = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", missing="skip")],
        )
    assert _DIFFERS_BY_ARM in warning_codes(rec)
    uf_skip = nw.from_native(_unit_rows(src_skip, _metric(src_skip, "revenue")), eager_only=True)
    assert uf_skip.shape[0] == 150


def test_unit_frame_ratio_missing_policy_covers_the_denominator_too() -> None:
    """A ratio metric's missing='zero'/'skip' apply to y_den exactly as to
    y (mirroring _apply_spec_missing's combined predicate) - a null
    denominator is not a hole the numerator's policy alone papers over."""
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(6)],
            "variant": ["control", "control", "control", "treatment", "treatment", "treatment"],
            "revenue": [10.0, None, 30.0, 40.0, 50.0, 60.0],
            "orders": [2.0, 3.0, None, 4.0, 5.0, 6.0],
        }
    )
    src_zero = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="aov",
                type="ratio",
                numerator="revenue",
                denominator="orders",
                missing="zero",
            )
        ],
    )
    uf_zero = nw.from_native(_unit_rows(src_zero, _metric(src_zero, "aov")), eager_only=True)
    assert uf_zero.shape[0] == 6
    assert uf_zero["y"].is_null().sum() == 0
    assert float(uf_zero["y"].sum()) == 190.0
    assert uf_zero["y_den"].is_null().sum() == 0
    assert float(uf_zero["y_den"].sum()) == 20.0

    # Observed 1/3 control vs 3/3 treatment: chi-square 3.0 on 1 df, p = 0.083.
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        src_skip = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(
                    name="aov",
                    type="ratio",
                    numerator="revenue",
                    denominator="orders",
                    missing="skip",
                )
            ],
        )
    assert _DIFFERS_BY_ARM not in warning_codes(recorded)
    uf_skip = nw.from_native(_unit_rows(src_skip, _metric(src_skip, "aov")), eager_only=True)
    # 2 of 6 rows skip: one null revenue, one null orders (distinct rows).
    assert uf_skip.shape[0] == 4


def test_terse_quantile_redirects_to_explicit_metricspec() -> None:
    """The terse form cannot carry quantile= - the message must point at
    the explicit MetricSpec form, exactly as the retention redirect does,
    instead of claiming quantile is an unknown type."""
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            _arrow_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "quantile"},
        )
    assert exc_info.value.code == "frame.metric_type_quantile"
    assert exc_info.value.context["name"] == "revenue"


def test_control_not_found_on_empty_frame_names_emptiness() -> None:
    table = pa.table(
        {
            "user_id": pa.array([], type=pa.string()),
            "variant": pa.array([], type=pa.string()),
            "revenue": pa.array([], type=pa.float64()),
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert error.value.code == "source.frame.control_missing"


def test_summary_conversion_metric_must_be_binary(arrow_frame: pa.Table) -> None:
    """type='conversion' is estimator-identical to mean, so a magnitude
    column wearing the label ships a number whose NAME promises a rate.
    Refused at the chokepoint, naming type='mean' as the honest choice."""
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            arrow_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", type="conversion")],
        )
    assert error.value.code == "source.frame.conversion_not_binary"


def test_summary_conversion_accepts_bool_and_declared_zero_nulls() -> None:
    """True/False columns and nulls under a declared missing='zero' stay
    accepted - the binary check rules on observed values only."""
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "converted": [True, False, True, None],
        }
    )
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion", missing="zero")],
    )
    out = {r["group_id"]: r for r in _moments(src, _metric(src, "converted"), grain="total")}
    assert _sum_y(out["treatment"]) == pytest.approx(1.0)


def test_tz_aware_date_column_refused_with_day_grain_contract() -> None:
    """A tz-aware day column used to crash with a raw backend cast
    TypeError naming nothing; the refusal must name the column and the
    caller-owns-bucketing contract."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "variant": ["control", "control", "treatment", "treatment"],
            "day": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]
            ).tz_localize("US/Eastern"),
            "revenue": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(
            df,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert error.value.code == "source.panel.day_grain_columns"


def test_tz_aware_exposure_date_column_refused() -> None:
    pd = pytest.importorskip("pandas")
    naive_days = pd.to_datetime(["2025-01-01", "2025-01-02"] * 2)
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "variant": ["control", "control", "treatment", "treatment"],
            "day": naive_days,
            "exposed_on": pd.to_datetime(["2025-01-01"] * 4).tz_localize("UTC"),
            "revenue": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(
            df,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="revenue", window_days=2)],
            exposure_date="exposed_on",
        )
    assert error.value.code == "source.panel.day_grain_columns"


def test_null_unit_id_refused_on_summary() -> None:
    """A null unit id would fail the densification/fold-hash join on some
    backends and be silently manufactured or dropped on others - refused
    at the boundary, before any duplicate check runs."""
    table = pa.table(
        {
            "user_id": ["u1", None],
            "variant": ["control", "treatment"],
            "revenue": [1.0, 2.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            table, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
        )
    assert error.value.code == "source.frame.null_identity"


def test_null_unit_id_refused_on_panel() -> None:
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": ["u1", None],
            "variant": ["control", "treatment"],
            "day": [base, base],
            "revenue": [1.0, 2.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert error.value.code == "source.frame.null_identity"


def test_null_date_refused_on_panel() -> None:
    """A null day axis value would silently zero-fill on pandas, raise a
    bare backend TypeError on polars, or a bare ArrowInvalid on pyarrow -
    refused at the boundary instead, naming the date column."""
    table = pa.table(
        {
            "user_id": ["u1", "u1"],
            "variant": ["control", "control"],
            "day": pa.array([date(2025, 1, 1), None]),
            "revenue": [1.0, 2.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
        )
    assert error.value.code == "source.frame.null_identity"


def test_infinite_metric_value_refused_on_summary() -> None:
    """+-inf is neither null nor NaN, so it survives every missing-value
    check and then diverges by backend at the first sum (inf on pandas,
    NaN on polars/pyarrow) - refused up front instead, naming the
    offending column."""
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "revenue": [1.0, float("inf"), 3.0, 4.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            table, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
        )
    assert error.value.code == "source.frame.non_finite"


def test_infinite_denominator_value_refused_on_summary() -> None:
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "revenue": [1.0, 2.0, 3.0, 4.0],
            "orders": [1.0, float("-inf"), 1.0, 2.0],
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(name="aov", type="ratio", numerator="revenue", denominator="orders")
            ],
        )
    assert error.value.code == "source.frame.non_finite"


def _null_exposure_panel() -> tuple[pa.Table, dict[str, Any]]:
    """5 units, u5 (treatment) with a null exposure date on every row;
    previously it silently vanished from windowed moments (n=4) while
    unit_counts()/srm() still counted it (n=5)."""
    base = date(2025, 1, 1)
    rows: list[tuple[str, str, date, date | None, float]] = []
    units = [
        ("u1", "control", base),
        ("u2", "control", base),
        ("u3", "treatment", base),
        ("u4", "treatment", base),
        ("u5", "treatment", None),
    ]
    for u, g, exp in units:
        for k in range(3):
            rows.append((u, g, base + timedelta(days=k), exp, float(k + 1)))
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue"],
                list(zip(*rows, strict=True)),
                strict=True,
            )
        )
    )
    kwargs: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "date": "day",
        "control": "control",
        "metrics": [MetricSpec(name="revenue", window_days=2)],
        "exposure_date": "exposed_on",
        "observation_end": date(2025, 1, 10),
    }
    return table, kwargs


def test_null_exposure_date_refuses_naming_the_exclude_knob() -> None:
    table, kwargs = _null_exposure_panel()
    with pytest.raises(InvalidRequestError) as error:
        from_unit_panel(table, **kwargs)
    assert error.value.code == "source.panel.null_exposure"


def test_null_exposure_date_excluded_with_accounting() -> None:
    """on_unassigned='exclude': the anchorless unit leaves the moments AND
    the unit counts together, surfaced once under '(unassigned)' - one
    unusable-unit concept, consistent between moments and the SRM check."""
    table, kwargs = _null_exposure_panel()
    src = from_unit_panel(table, **kwargs, on_unassigned="exclude")
    assert _counts(src, "unit") == {"control": 2, "treatment": 2, "(unassigned)": 1}
    out = {r["group_id"]: r for r in _moments(src, _metric(src, "revenue"), grain="total")}
    assert out["treatment"]["n"] == 2
    assert out["control"]["n"] == 2
    # window [0, 2) keeps days 1 and 2 of each unit: 1.0 + 2.0 per unit
    assert _sum_y(out["treatment"]) == pytest.approx(6.0)


def test_summary_null_exposure_excluded_with_accounting() -> None:
    """The unit summary counts a null-exposure exclusion like the panel does."""
    table = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "variant": ["control", "control", "treatment", "treatment", "treatment"],
            "exposed_on": [date(2025, 1, 1)] * 4 + [None],
            "revenue": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        exposure_date="exposed_on",
        on_unassigned="exclude",
    )
    assert _counts(src, "unit") == {"control": 2, "treatment": 2, "(unassigned)": 1}


def test_densified_cells_counts_declared_zero_null_fills() -> None:
    """A genuine in-row null under missing='zero' is manufactured into a
    zero by densification exactly like an absent (unit, day) cell - the
    counter must include both, not just the synthesised spine rows."""
    table = pa.table(
        {
            "user_id": ["u1", "u1", "u2"],
            "variant": ["control", "control", "treatment"],
            "day": ["d1", "d2", "d1"],
            "revenue": [1.0, None, 2.0],
        }
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", missing="zero")],
    )
    # 1 synthesised (u2, d2) cell + 1 genuine-null (u1, d2) cell
    assert src.densified_cells == 2


def test_densified_cells_counts_only_eligible_cells() -> None:
    """Only a unit's own days from its exposure through the last day can be zero filled.

    Every unit has a row on each day from its exposure (day u % 3) through day 6, so no
    eligible cell is missing; removing one of those rows leaves exactly one.
    """
    first = date(2026, 2, 1)
    rows = [
        {
            "unit": u,
            "arm": "control" if u % 2 else "treatment",
            "day": first + timedelta(days=k),
            "exposed": first + timedelta(days=u % 3),
            "value": 1.0,
        }
        for u in range(60)
        for k in range(u % 3, 7)
    ]

    def densified(table_rows: list[dict[str, Any]]) -> int:
        return from_unit_panel(
            pa.Table.from_pylist(table_rows),
            unit="unit",
            group="arm",
            date="day",
            control="control",
            exposure_date="exposed",
            metrics={"value": "mean"},
        ).densified_cells

    assert densified(rows) == 0
    assert densified(rows[:3] + rows[4:]) == 1


def test_densified_cells_memory_does_not_grow_with_distinct_exposure_dates() -> None:
    """Counting eligible cells reads each day and each exposure date once.

    Two panels with the same rows and days differ only in how many distinct exposure
    dates they hold (one, or one per day). Pairing every exposure date with every day
    would allocate 1500 x 1500 rows (18 MB per 8-byte column) for the second.
    """
    import tracemalloc

    import pandas as pd

    n = 1500

    def peak_bytes(exposed: list[int]) -> int:
        frame = pd.DataFrame(
            {
                "unit": range(n),
                "arm": ["control" if u % 2 else "treatment" for u in range(n)],
                "day": range(n),
                "exposed": exposed,
                "value": [1.0] * n,
            }
        )
        tracemalloc.start()
        try:
            from_unit_panel(
                frame,
                unit="unit",
                group="arm",
                date="day",
                control="control",
                exposure_date="exposed",
                metrics={"value": "mean"},
            )
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    peak_bytes([0] * n)  # warm imports and caches
    assert peak_bytes(list(range(n))) - peak_bytes([0] * n) < 4 * 2**20


def test_single_unit_arm_error_names_metric_and_arm() -> None:
    """A one-unit treatment arm within the declared roster is refused at estimation time.
    The moment-grain read is legitimate; the arm's only cell is an unavailable row whose
    failure names the metric and arm."""
    rows = [
        *(row for row in _ROWS if row[1] == "control"),
        next(row for row in _ROWS if row[1] == "treatment"),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(_COLUMNS, cols, strict=True)))
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=_DESIGN,
    )
    with pytest.warns(IncrementWarning):
        (row,) = readouts.run(src)
    assert (row.metric, row.group_id) == ("revenue", "treatment")
    assert row.failure_code == "estimation.armstats.arm_stats.least_compute_metric"
    assert row.failure_context is not None
    assert row.failure_context["metric"] == "revenue"
    assert row.failure_context["group_id"] == "treatment"


def test_windowed_encouragement_panel_offers_asof_but_refuses_total() -> None:
    base = date(2025, 1, 1)
    rows = []
    for unit, variant, revenue in (
        ("c1", "control", 1.0),
        ("c2", "control", 3.0),
        ("t1", "treatment", 2.0),
        ("t2", "treatment", 6.0),
    ):
        for offset in range(5):
            rows.append(
                {
                    "user_id": unit,
                    "variant": variant,
                    "day": base + timedelta(days=offset),
                    "exposed_on": base,
                    "clicked": float(variant == "treatment" and offset < 1),
                    "revenue": revenue if offset < 2 else 1000.0,
                }
            )
    src = from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=2)],
        uptake="clicked",
        design=_encouragement_design(window_days=3),
        exposure_date="exposed_on",
    )
    metric = src.context.metrics[0]

    assert _moments(src, metric, grain="asof")
    with pytest.raises(CapabilityError) as exc_grain:
        _moments(src, metric, grain="total")
    assert exc_grain.value.code == "source.frame.windowed_encouragement_total"


def test_encouragement_retention_remains_refused_at_construction() -> None:
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": ["c1", "c1", "t1", "t1"],
            "variant": ["control", "control", "treatment", "treatment"],
            "day": [base, base + timedelta(days=1)] * 2,
            "exposed_on": [base] * 4,
            "clicked": [0.0, 0.0, 1.0, 0.0],
            "returned": [0.0, 1.0, 0.0, 1.0],
        }
    )
    with pytest.raises(CapabilityError) as exc:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="returned", type="retention", threshold_days=(1, 3))],
            uptake="clicked",
            design=_encouragement_design(window_days=3),
            exposure_date="exposed_on",
        )
    assert exc.value.code == "readout.encouragement.retention"


@pytest.mark.parametrize("with_encouragement", [True, False])
@pytest.mark.parametrize(
    ("band", "code"),
    [
        ((5, 3), "definition.retention.threshold_days_upper_exceeds_lower"),
        ((-1, 4), "definition.retention.threshold_days_lower_bound_non_negative"),
    ],
)
def test_mapping_metric_with_invalid_retention_band_refuses_with_the_definition_code(
    with_encouragement: bool, band: tuple[int, int], code: str
) -> None:
    """A raw mapping metric reaches the same canonical refusal as ``MetricSpec``,
    with or without an encouragement design (whose retention capability refusal
    must not mask an invalid declaration)."""
    base = date(2025, 1, 1)
    table = pa.table(
        {
            "user_id": ["c1", "c1", "t1", "t1"],
            "variant": ["control", "control", "treatment", "treatment"],
            "day": [base, base + timedelta(days=1)] * 2,
            "exposed_on": [base] * 4,
            "clicked": [0.0, 0.0, 1.0, 0.0],
            "returned": [0.0, 1.0, 0.0, 1.0],
        }
    )
    extra: dict[str, Any] = (
        {"uptake": "clicked", "design": _encouragement_design(window_days=3)}
        if with_encouragement
        else {}
    )
    with pytest.raises(DefinitionError) as exc:
        from_unit_panel(
            table,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[{"name": "returned", "type": "retention", "threshold_days": band}],
            exposure_date="exposed_on",
            **extra,
        )
    assert exc.value.code == code


def _windowed_encouragement_source(
    *, outcome_window_days: int, uptake_window_days: int, periods: int
):
    base = date(2025, 1, 1)
    rows = []
    for unit, variant, value, click_offset in (
        ("c1", "control", 1.0, None),
        ("c2", "control", 3.0, None),
        ("t1", "treatment", 2.0, 2),
        ("t2", "treatment", 6.0, 0),
    ):
        for offset in range(periods):
            rows.append(
                {
                    "user_id": unit,
                    "variant": variant,
                    "day": base + timedelta(days=offset),
                    "exposed_on": base,
                    "clicked": float(offset == click_offset),
                    "revenue": value if offset < outcome_window_days else 1000.0,
                }
            )
    src = from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=outcome_window_days)],
        uptake="clicked",
        design=_encouragement_design(window_days=uptake_window_days),
        exposure_date="exposed_on",
    )
    return src, src.context.metrics[0], base


def test_asof_encouragement_completed_windows_gate_later_edge() -> None:
    src, metric, base = _windowed_encouragement_source(
        outcome_window_days=2,
        uptake_window_days=3,
        periods=5,
    )

    rows = _moments(src, metric, grain="asof", completed_windows_only=True)

    assert min(row["ds"] for row in rows) == base + timedelta(days=3)
    treatment = [row for row in rows if row["group_id"] == "treatment"]
    assert {row["sum_d"] for row in treatment} == {2.0}
    assert {row["ref_y"] for row in treatment} == {8.0}


def test_censoring_warning_attributed_to_caller_not_engine_internals() -> None:
    """The warning's source line must survive changes to the internal call
    chain: it is attributed to the first frame outside the package, not a
    hardcoded number of frames up."""
    src = _fence_panel(date(2025, 1, 13))
    metric = _metric(src, "revenue")
    with pytest.warns(IncrementWarning) as record:
        from increment.evidence import MomentsRequest, evidence_provider, request_evidence

        request = MomentsRequest(metric, grain="total")
        request_evidence(evidence_provider(src), request)
    (warning,) = [
        r
        for r in record
        if isinstance(r.message, IncrementWarning)
        and r.message.code == "frame.censoring.dropped_units"
    ]
    assert warning.filename == __file__


def test_missing_impute_is_refused_by_every_frame_constructor_with_one_code() -> None:
    spec = {"name": "revenue", "missing": "impute"}
    constructors = {
        "from_unit_summary": lambda: from_unit_summary(
            _arrow_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[spec],
        ),
        "from_unit_panel": lambda: from_unit_panel(
            _panel_table(),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[spec],
        ),
    }
    for ingress, construct in constructors.items():
        raised = refused(IMPUTE_POOLED_MEAN, f"{ingress}-missing-impute-refused", construct)
        assert isinstance(raised, InvalidRequestError)
        assert raised.context["metric"] == "revenue"
