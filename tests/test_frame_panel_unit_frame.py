import numpy as np
import polars as pl
import pytest

from increment import AdjustmentSet, Analysis, MetricSpec, Observational, Winsorization
from increment.errors import CodedError
from tests.analysis_factory import _native_source, lift_rows


def _panel_with_covariate(n=40, *, vary_within_unit=False):
    rng = np.random.default_rng(4)
    tenure = {f"u{i}": float(rng.normal(100, 15)) for i in range(n)}
    group = {f"u{i}": ("control" if i % 2 == 0 else "treatment") for i in range(n)}
    rows = []
    for i in range(n):
        uid = f"u{i}"
        for day in range(2):
            rows.append(
                {
                    "user_id": uid,
                    "variant": group[uid],
                    "date": f"2025-01-0{day + 1}",
                    "exposure_date": "2025-01-01",
                    "revenue": 5.0 + 0.1 * i + day,
                    "tenure": tenure[uid] + (day if vary_within_unit else 0.0),
                }
            )
    return pl.DataFrame(rows), tenure, group


def test_unit_panel_unit_frame_serves_constant_within_unit_covariate():
    df, tenure, _group = _panel_with_covariate()
    analysis = Analysis.from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="date",
        metrics={"revenue": "mean"},
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        ),
    )
    table = _native_source(analysis).unit_frame(
        _native_source(analysis).context.metrics[0], covariates=["tenure"]
    )
    rows = (
        {r["unit_id"]: r["tenure"] for r in table.to_pylist()}
        if hasattr(table, "to_pylist")
        else {r["unit_id"]: r["tenure"] for r in table.to_dicts()}
    )
    assert rows == pytest.approx(tenure)


def test_unit_panel_unit_frame_refuses_covariate_varying_within_unit():
    df, _tenure, _group = _panel_with_covariate(vary_within_unit=True)
    analysis = Analysis.from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="date",
        metrics={"revenue": "mean"},
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        ),
    )
    with pytest.raises(CodedError) as raised:
        _native_source(analysis).unit_frame(
            _native_source(analysis).context.metrics[0], covariates=["tenure"]
        )
    assert raised.value.code == "frame.frame_panel.unit_covariate_varies"


def test_unordered_categorical_panel_matches_summary_rows():
    import pandas as pd

    from tests.categorical_cases import assert_rows_match, categorical_units

    summary_frame = pd.DataFrame(categorical_units(400, 23))
    summary_frame["region"] = pd.Categorical(
        summary_frame["region"], categories=["west", "unused", "east", "north"], ordered=False
    )
    panel_frame = pd.concat(
        [summary_frame.assign(date=day, revenue=summary_frame["revenue"] / 2) for day in (1, 2)],
        ignore_index=True,
    )
    design = Observational(
        control_group="C", adjustment=AdjustmentSet(covariates=("spend", "region"))
    )
    panel = Analysis.from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="date",
        metrics={"revenue": "mean"},
        design=design,
    )
    summary = Analysis.from_unit_summary(
        summary_frame,
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=design,
    )
    assert_rows_match(
        [row.model_dump(mode="json") for row in lift_rows(summary.run())],
        [row.model_dump(mode="json") for row in lift_rows(panel.run())],
    )


def _is_missing(value) -> bool:
    return value is None or value != value


def test_unit_panel_unit_frame_serves_all_null_covariates_as_missing_on_pandas():
    """A unit whose covariate is null on every one of its rows is constant
    and missing, never "varying": both a numeric NaN and a categorical
    string level are served as a missing value on a pandas frame, whose
    null-vs-null comparison reads as unequal."""
    import pandas as pd

    rows = []
    for i in range(12):
        uid = f"u{i}"
        for day in range(2):
            rows.append(
                {
                    "user_id": uid,
                    "variant": "control" if i % 2 == 0 else "treatment",
                    "date": f"2025-01-0{day + 1}",
                    "revenue": 5.0 + 0.1 * i + day,
                    "tenure": float("nan") if uid == "u5" else 100.0 + i,
                    "region": None if uid == "u3" else ("east" if i % 3 else "west"),
                }
            )
    frame = pd.DataFrame(rows)
    frame["region"] = pd.Categorical(frame["region"], ordered=False)
    analysis = Analysis.from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="date",
        metrics={"revenue": "mean"},
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("tenure", "region"), missing="impute-indicator"),
        ),
    )
    source = _native_source(analysis)
    table = source.unit_frame(source.context.metrics[0], covariates=["tenure", "region"])
    served = {r["unit_id"]: r for r in _frame_records(table)}
    assert set(served) == {f"u{i}" for i in range(12)}
    assert _is_missing(served["u3"]["region"]) and served["u3"]["tenure"] == pytest.approx(103.0)
    assert _is_missing(served["u5"]["tenure"]) and served["u5"]["region"] == "east"
    assert served["u4"]["region"] == "east" and served["u6"]["region"] == "west"


def test_unit_panel_unit_frame_still_refuses_windowed_metric():
    df, _tenure, _group = _panel_with_covariate()
    analysis = Analysis.from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="date",
        exposure_date="exposure_date",
        metrics=[{"name": "revenue", "type": "mean", "window_days": 1}],
        control="control",
    )
    with pytest.raises(CodedError) as raised:
        _native_source(analysis).unit_frame(_native_source(analysis).context.metrics[0])
    assert raised.value.code == "source.frame.unit_frame_panel"


def test_unit_panel_unit_frame_serves_unwindowed_quantile_metric_matching_unit_summary():
    """An unwindowed quantile metric's per-unit value is the same
    per-unit total the panel's collapse already produces for mean/ratio --
    the order statistic is taken across units downstream, not within this
    per-unit reduction, so an unwindowed quantile is no longer refused."""
    df, _tenure, _group = _panel_with_covariate()
    spec = {"name": "revenue", "type": "quantile", "quantile": 0.5}
    panel = Analysis.from_unit_panel(
        df, unit="user_id", group="variant", date="date", metrics=[spec], control="control"
    )
    summary_df = df.group_by(["user_id", "variant"]).agg(pl.col("revenue").sum())
    summary = Analysis.from_unit_summary(
        summary_df, unit="user_id", group="variant", metrics=[spec], control="control"
    )
    panel_rows = {
        r["unit_id"]: r["y"]
        for r in _frame_records(
            _native_source(panel).unit_frame(_native_source(panel).context.metrics[0])
        )
    }
    summary_rows = {
        r["unit_id"]: r["y"]
        for r in _frame_records(
            _native_source(summary).unit_frame(_native_source(summary).context.metrics[0])
        )
    }
    assert panel_rows == pytest.approx(summary_rows)


def test_unit_panel_quantile_metric_cannot_declare_window_days():
    """A quantile metric can never carry `window_days` on the frame path
    -- refused at `MetricSpec` construction, well before `unit_frame` is
    ever reached, so there is no windowed-quantile case for `unit_frame`
    to refuse."""
    df, _tenure, _group = _panel_with_covariate()
    with pytest.raises(CodedError) as raised:
        Analysis.from_unit_panel(
            df,
            unit="user_id",
            group="variant",
            date="date",
            exposure_date="exposure_date",
            metrics=[{"name": "revenue", "type": "quantile", "quantile": 0.5, "window_days": 1}],
            control="control",
        )
    assert raised.value.code == "frame.metric.window_days_supported"


def _frame_records(table):
    import narwhals as nw

    return nw.from_native(table, eager_only=True).rows(named=True)


def _confounded_unit_data(n=60, *, outlier_unit=None, outlier_value=None, null_units=()):
    rng = np.random.default_rng(7)
    units = [f"u{i}" for i in range(n)]
    tenure = {u: float(rng.normal(100, 15)) for u in units}
    propensity = {u: 1.0 / (1.0 + np.exp(-(tenure[u] - 100) / 20.0)) for u in units}
    group = {u: ("treatment" if rng.random() < propensity[u] else "control") for u in units}
    revenue: dict[str, float | None] = {
        u: 5.0
        + 0.05 * tenure[u]
        + (3.0 if group[u] == "treatment" else 0.0)
        + float(rng.normal(0, 1))
        for u in units
    }
    if outlier_unit is not None:
        revenue[outlier_unit] = outlier_value
    for u in null_units:
        revenue[u] = None
    return units, tenure, group, revenue


def _panel_and_summary(units, tenure, group, revenue, spec, design):
    n = len(units)
    panel_df = pl.DataFrame(
        {
            "user_id": units,
            "variant": [group[u] for u in units],
            "date": ["2025-01-01"] * n,
            "revenue": [revenue[u] for u in units],
            "tenure": [tenure[u] for u in units],
        }
    )
    panel = Analysis.from_unit_panel(
        panel_df, unit="user_id", group="variant", date="date", metrics=[spec], design=design
    )
    summary = Analysis.from_unit_summary(
        panel_df.drop("date"), unit="user_id", group="variant", metrics=[spec], design=design
    )
    return panel, summary


def test_unit_panel_unit_frame_applies_winsorization_matching_unit_summary():
    """The panel's collapsed unit total must be winsorized the same way
    moments(grain="total") already winsorizes it -- not the raw sum."""
    from increment import IdentificationGate

    units, tenure, group, revenue = _confounded_unit_data(outlier_unit="u0", outlier_value=500.0)
    spec = MetricSpec(name="revenue", type="mean", winsorization=Winsorization(upper_value=50.0))
    design = Observational(
        control_group="control",
        adjustment=AdjustmentSet(covariates=("tenure",)),
        gate=IdentificationGate(overlap="trim"),
    )
    panel, summary = _panel_and_summary(units, tenure, group, revenue, spec, design)

    metric = _native_source(panel).context.metrics[0]
    panel_rows = {
        r["unit_id"]: r["y"] for r in _frame_records(_native_source(panel).unit_frame(metric))
    }
    summary_rows = {
        r["unit_id"]: r["y"] for r in _frame_records(_native_source(summary).unit_frame(metric))
    }
    assert panel_rows == pytest.approx(summary_rows)
    # The outlier must actually be clipped, not silently passed through raw.
    assert panel_rows["u0"] == pytest.approx(50.0)

    panel_result = {r.group_id: r for r in lift_rows(panel.run())}
    summary_result = {r.group_id: r for r in lift_rows(summary.run())}
    assert set(panel_result) == set(summary_result)
    for group_id, row in panel_result.items():
        oracle = summary_result[group_id]
        for field in ("value", "lb", "ub"):
            assert getattr(row.lift, field) == pytest.approx(getattr(oracle.lift, field), rel=1e-9)


def test_unit_panel_unit_frame_applies_missing_zero_matching_unit_summary():
    """The panel's collapsed unit total must apply missing="zero" the same
    way moments(grain="total") already applies it. missing="drop" itself is
    refused outright at panel construction (densification already
    zero-fills every absent/null cell, so a "dropped" row would silently
    mean zero anyway) -- "zero" is the only non-default policy a
    panel-backed metric can declare, so it is the one this parity check
    exercises."""
    from increment import IdentificationGate

    units, tenure, group, revenue = _confounded_unit_data(null_units=("u1", "u5"))
    spec = MetricSpec(name="revenue", type="mean", missing="zero")
    design = Observational(
        control_group="control",
        adjustment=AdjustmentSet(covariates=("tenure",)),
        gate=IdentificationGate(overlap="trim"),
    )
    panel, summary = _panel_and_summary(units, tenure, group, revenue, spec, design)

    metric = _native_source(panel).context.metrics[0]
    panel_rows = {
        r["unit_id"]: r["y"] for r in _frame_records(_native_source(panel).unit_frame(metric))
    }
    summary_rows = {
        r["unit_id"]: r["y"] for r in _frame_records(_native_source(summary).unit_frame(metric))
    }
    assert panel_rows["u1"] == pytest.approx(0.0)
    assert panel_rows["u5"] == pytest.approx(0.0)
    assert panel_rows == pytest.approx(summary_rows)

    panel_result = {r.group_id: r for r in lift_rows(panel.run())}
    summary_result = {r.group_id: r for r in lift_rows(summary.run())}
    assert set(panel_result) == set(summary_result)
    for group_id, row in panel_result.items():
        oracle = summary_result[group_id]
        for field in ("value", "lb", "ub"):
            assert getattr(row.lift, field) == pytest.approx(getattr(oracle.lift, field), rel=1e-9)


def test_unit_panel_unit_frame_refuses_covariate_null_on_some_rows_only():
    """A unit whose covariate is null on some of its own rows and set on
    others is not a stable per-unit value either -- must refuse the same
    way a unit with two disagreeing non-null values refuses."""
    n = 10
    rng = np.random.default_rng(9)
    tenure = {f"u{i}": float(rng.normal(100, 15)) for i in range(n)}
    group = {f"u{i}": ("control" if i % 2 == 0 else "treatment") for i in range(n)}
    rows = []
    for i in range(n):
        uid = f"u{i}"
        for day in range(2):
            # u0's covariate is null on its first day and set on its second --
            # every other unit's covariate is constant across both its days.
            covariate = None if (uid == "u0" and day == 0) else tenure[uid]
            rows.append(
                {
                    "user_id": uid,
                    "variant": group[uid],
                    "date": f"2025-01-0{day + 1}",
                    "revenue": 5.0 + 0.1 * i + day,
                    "tenure": covariate,
                }
            )
    df = pl.DataFrame(rows)
    analysis = Analysis.from_unit_panel(
        df,
        unit="user_id",
        group="variant",
        date="date",
        metrics={"revenue": "mean"},
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        ),
    )
    with pytest.raises(CodedError) as raised:
        _native_source(analysis).unit_frame(
            _native_source(analysis).context.metrics[0], covariates=["tenure"]
        )
    assert raised.value.code == "frame.frame_panel.unit_covariate_varies"
    units_ctx = raised.value.context["units"]
    assert isinstance(units_ctx, (list, tuple))
    assert "u0" in units_ctx
