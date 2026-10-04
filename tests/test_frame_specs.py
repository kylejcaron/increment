"""Tests for the dataframe entry point (``increment/frame.py``).

The load-bearing tests are ``test_moments_match_group_summary`` and
``test_daily_moments_match_daily_group_summary``: they assert the narwhals
moments equal the real ibis builders column for column. Both import ibis;
``increment/frame.py`` must not.
"""

from __future__ import annotations

import importlib.util
import pickle
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import narwhals as nw
import numpy as np
import pyarrow as pa
import pytest

import increment._metric_specs as metric_specs
import increment.frame as frame
from increment import readouts
from increment.errors import CapabilityError, DefinitionError, InvalidRequestError
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.frame import (
    FramePanelSource,
    FrameTotalsSource,
    MetricSpec,
    coerce_metrics,
    from_unit_panel,
    from_unit_summary,
    synthesise_metric,
)
from increment.semantics.design import Randomized
from increment.semantics.models import MeanMetric, Metric, RatioMetric, RetentionMetric

_DESIGN = Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
)


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


def _backend_frames() -> list[tuple[str, Any]]:
    """One frame per installed backend, for cross-backend equivalence."""
    frames: list[tuple[str, Any]] = [("pyarrow", _arrow_table())]
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        frames.append(("pandas", pd.DataFrame(_ROWS, columns=_COLUMNS)))
    if importlib.util.find_spec("polars") is not None:
        import polars as pl

        frames.append(("polars", pl.from_arrow(_arrow_table())))
    return frames


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


@pytest.mark.parametrize(
    ("spec", "capability", "value"),
    [
        (MetricSpec(name="revenue", window_days=7), "window_days", 7),
        (
            MetricSpec(name="d7", type="retention", threshold_days=7),
            "type=retention",
            "retention",
        ),
    ],
)
def test_summary_refuses_time_dependent_metrics_with_structured_context(
    spec: MetricSpec, capability: str, value: int | str
) -> None:
    with pytest.raises(CapabilityError) as exc:
        from_unit_summary(
            _arrow_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[spec],
        )
    assert exc.value.code == "source.frame.constructor"
    assert exc.value.context["metric"] == spec.name
    assert exc.value.context["capability"] == capability
    assert exc.value.context["value"] == value
    restored = pickle.loads(pickle.dumps(exc.value))
    assert restored.code == exc.value.code
    assert restored.context == exc.value.context


# Constructs (``MetricSpec`` does not bound a mean's ``window_days``) but fails
# in metric synthesis (``MeanMetric.window_days`` is ``ge=1``): a witness that
# the capability guards run before any synthesis.
_SYNTHESIS_ONLY_FAILURE = MetricSpec(name="bad_window", window_days=0)


def test_from_unit_summary_retention_capability_error_beats_synthesis_validation_error() -> None:
    """``_reject_windowed_specs`` must run before any metric synthesis, so a
    valid retention ``MetricSpec`` that is simply unsupported on this shape
    still raises the guard's own ``CapabilityError`` -- not an error surfaced
    from deep inside the decision compiler's eager metric synthesis.
    """
    with pytest.raises(CapabilityError) as exc:
        from_unit_summary(
            _arrow_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(name="d7", type="retention", threshold_days=(1, 3)),
                _SYNTHESIS_ONLY_FAILURE,
            ],
        )
    assert exc.value.code == "source.frame.constructor"
    assert exc.value.context["metric"] == "d7"


def test_from_unit_panel_covariate_refusal_beats_synthesis_validation_error() -> None:
    """Same ordering guarantee on the panel path: ``_reject_covariates``
    runs before any metric synthesis, so a windowed covariate metric declared
    alongside a retention spec still raises the covariate
    ``frame.validation.from_unit_panel`` refusal, not an error from
    synthesising the other (unrelated) spec first.
    """
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_panel(
            _panel_table(),
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[
                MetricSpec(name="revenue", covariate="pre_revenue", window_days=7),
                MetricSpec(name="d7", type="retention", threshold_days=(1, 3)),
                _SYNTHESIS_ONLY_FAILURE,
            ],
        )
    assert exc_info.value.code == "frame.validation.from_unit_panel"
    assert exc_info.value.context["covered"] == ("revenue",)


def test_metric_spec_accepts_window_days() -> None:
    spec = MetricSpec(name="revenue", window_days=7)
    assert spec.window_days == 7


def test_metric_spec_accepts_mean_winsorization() -> None:
    spec = MetricSpec(name="revenue", winsorization={"upper_value": 500.0})
    metric = synthesise_metric(spec)
    assert isinstance(metric, MeanMetric)
    assert metric.winsorization is not None
    assert metric.winsorization.upper_value == 500.0


def test_summary_winsorization_uses_one_pooled_cutoff_across_arms() -> None:
    table = pa.table(
        {
            "user_id": ["c1", "c2", "t1", "t2"],
            "variant": ["control", "control", "treatment", "treatment"],
            "revenue": [1.0, 100.0, 2.0, 200.0],
        }
    )
    source = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                winsorization={"upper_value": 10.0},
            )
        ],
    )
    rows = {row["group_id"]: row for row in source.raw_moments}
    assert rows["control"]["ref_y"] == pytest.approx(5.5)
    assert rows["treatment"]["ref_y"] == pytest.approx(6.0)
    assert rows["control"]["winsor_upper_bound"] == 10.0
    assert rows["treatment"]["winsor_n_upper"] == 1


def test_unit_frame_exposes_winsorized_outcomes() -> None:
    table = pa.table(
        {
            "user_id": ["c1", "c2", "t1", "t2"],
            "variant": ["control", "control", "treatment", "treatment"],
            "revenue": [1.0, 100.0, 2.0, 200.0],
        }
    )
    source = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", winsorization={"upper_value": 10.0})],
    )
    metric = cast(MeanMetric, source.context.metrics[0])
    units = nw.from_native(source.unit_frame(metric), eager_only=True)
    values = sorted(float(value) for value in units["y"].to_list())
    assert values == [1.0, 2.0, 10.0, 10.0]


def test_summary_percentile_winsorization_uses_linear_interpolation() -> None:
    table = pa.table(
        {
            "user_id": ["c1", "c2", "t1", "t2"],
            "variant": ["control", "control", "treatment", "treatment"],
            "revenue": [1.0, 2.0, 100.0, 200.0],
        }
    )
    source = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", winsorization={"upper_percentile": 0.75})],
    )
    rows = {row["group_id"]: row for row in source.raw_moments}
    assert rows["control"]["ref_y"] == pytest.approx(1.5)
    assert rows["treatment"]["ref_y"] == pytest.approx(112.5)
    assert rows["treatment"]["winsor_upper_bound"] == pytest.approx(125.0)
    assert rows["treatment"]["winsor_n_upper"] == 1


def test_panel_total_winsorization_runs_after_unit_collapse() -> None:
    table = pa.table(
        {
            "user_id": ["c1", "c1", "c2", "t1", "t2"],
            "variant": ["control", "control", "control", "treatment", "treatment"],
            "day": [1, 2, 1, 1, 1],
            "revenue": [1.0, 9.0, 100.0, 2.0, 200.0],
        }
    )
    source = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", winsorization={"upper_value": 10.0})],
    )
    metric = cast(MeanMetric, source.context.metrics[0])
    rows = {row["group_id"]: row for row in source.moments(metric)}
    assert rows["control"]["ref_y"] == pytest.approx(10.0)
    assert rows["treatment"]["ref_y"] == pytest.approx(6.0)
    assert rows["control"]["winsor_n_upper"] == 1


@pytest.mark.parametrize(
    "backend,table", _backend_frames(), ids=lambda value: value if isinstance(value, str) else None
)
def test_winsorization_matches_across_dataframe_backends(backend: str, table: Any) -> None:
    source = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", winsorization={"upper_value": 10.0})],
    )
    rows = {row["group_id"]: row for row in source.raw_moments}
    assert rows["control"]["ref_y"] == pytest.approx(10.0)
    assert rows["treatment"]["ref_y"] == pytest.approx(10.0)
    assert backend in {"pyarrow", "pandas", "polars"}


def test_metric_spec_rejects_winsorization_on_non_mean() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        MetricSpec(
            name="converted",
            type="conversion",
            winsorization={"upper_value": 1.0},
        )
    assert exc_info.value.code == "frame.metric.winsorization_applies_type"
    assert exc_info.value.context["name"] == "converted"


def test_metric_spec_retention_band_forms() -> None:
    assert MetricSpec(name="d7", type="retention", threshold_days=7).threshold_days == 7
    banded = MetricSpec(name="d7", type="retention", threshold_days=(7, 14))
    assert banded.threshold_days == (7, 14)


@pytest.mark.parametrize(
    ("band", "code"),
    [
        ((3, 3), "definition.retention.threshold_days_upper_exceeds_lower"),
        ((5, 3), "definition.retention.threshold_days_upper_exceeds_lower"),
        ((-1, 4), "definition.retention.threshold_days_lower_bound_non_negative"),
        (-1, "definition.retention.threshold_days_non_negative"),
    ],
)
def test_metric_spec_refuses_an_invalid_retention_band_at_construction(
    band: int | tuple[int, int], code: str
) -> None:
    with pytest.raises(DefinitionError) as exc_info:
        MetricSpec(name="d7", type="retention", threshold_days=band)
    assert exc_info.value.code == code
    assert exc_info.value.context["name"] == "d7"


@pytest.mark.parametrize("band", [True, (False, 3), (1, True)])
def test_metric_spec_refuses_bool_day_counts(band: object) -> None:
    with pytest.raises(DefinitionError) as exc_info:
        MetricSpec(name="d7", type="retention", threshold_days=band)
    assert exc_info.value.code == "definition.models.reject_bool"
    with pytest.raises(DefinitionError) as exc_info:
        MetricSpec(name="revenue", window_days=True)
    assert exc_info.value.code == "definition.models.reject_bool"


def test_retention_requires_threshold_days() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        MetricSpec(name="d7", type="retention")
    assert exc_info.value.code == "frame.metric.type_retention_needs"
    assert exc_info.value.context["name"] == "d7"


def test_threshold_days_requires_retention_type() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        MetricSpec(name="revenue", threshold_days=7)
    assert exc_info.value.code == "frame.metric.sets_threshold_days"
    assert exc_info.value.context["name"] == "revenue"


@pytest.mark.parametrize("window_days", [0, -1])
def test_retention_window_days_refusal_code_matches_across_constructors(window_days: int) -> None:
    with pytest.raises(DefinitionError) as spec_exc:
        MetricSpec(name="d7", type="retention", threshold_days=7, window_days=window_days)
    with pytest.raises(DefinitionError) as metric_exc:
        RetentionMetric(name="d7", entity="u", fact="f", threshold_days=7, window_days=window_days)
    assert spec_exc.value.code == metric_exc.value.code == "definition.retention.metric_window_days"


def test_window_days_rejected_on_retention_and_quantile() -> None:
    # band lives in threshold_days - models-layer validation, single-sourced
    with pytest.raises(DefinitionError) as exc_info:
        MetricSpec(name="d7", type="retention", threshold_days=7, window_days=14)
    assert exc_info.value.code == "definition.retention.metric_window_days"
    with pytest.raises(InvalidRequestError) as exc_info:
        MetricSpec(name="p50", type="quantile", quantile=0.5, window_days=7)
    assert exc_info.value.code == "frame.metric.window_days_supported"
    assert exc_info.value.context == {"name": "p50"}


def test_summary_constructor_refuses_windowed_and_retention_specs(arrow_frame: pa.Table) -> None:
    # a one-row-per-unit summary has no dates: nothing to window
    for spec in (
        MetricSpec(name="revenue", window_days=7),
        MetricSpec(name="d7", type="retention", threshold_days=7),
    ):
        with pytest.raises(CapabilityError) as exc:
            from_unit_summary(
                arrow_frame,
                unit="user_id",
                group="variant",
                control="control",
                metrics=[spec],
            )
        assert exc.value.code == "source.frame.constructor"


def test_capability_errors_never_silently_change_a_number(panel_frame: pa.Table) -> None:
    """No unsupported path degrades to a different number - it always raises."""
    with pytest.raises(ValueError):
        from_unit_panel(
            panel_frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics=[MetricSpec(name="revenue", window_days=7)],
        )


def test_metric_spec_quantile_requires_q() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        MetricSpec(name="latency", type="quantile")
    assert exc_info.value.code == "frame.metric.type_quantile_needs"
    assert exc_info.value.context["name"] == "latency"


def test_metric_spec_quantile_rejects_covariate() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        MetricSpec(name="latency", type="quantile", quantile=0.9, covariate="pre")
    assert exc_info.value.code == "frame.metric.cuped_does_apply"
    assert exc_info.value.context["name"] == "latency"


def test_metric_spec_quantile_field_rejected_on_non_quantile_type() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        MetricSpec(name="latency", type="mean", quantile=0.5)
    assert exc_info.value.code == "frame.metric.sets_quantile_but"
    assert exc_info.value.context == {"name": "latency", "type": "mean"}


def test_metric_spec_preferred_direction_defaults_to_none() -> None:
    """Unlike MetricIdentity.preferred_direction (defaults to 'increase'),
    MetricSpec must default to None and stay untracked -- a frame metric
    that never declares a direction must be distinguishable from one that
    explicitly chose 'increase'."""
    spec = MetricSpec(name="revenue")
    assert spec.preferred_direction is None
    assert "preferred_direction" not in spec.model_fields_set


def test_metric_spec_accepts_preferred_direction() -> None:
    spec = MetricSpec(name="latency", preferred_direction="decrease")
    assert spec.preferred_direction == "decrease"
    assert "preferred_direction" in spec.model_fields_set


_SYNTHESISE_SPEC_KWARGS: dict[str, dict[str, Any]] = {
    "mean": {"name": "revenue", "type": "mean"},
    "conversion": {"name": "converted", "type": "conversion"},
    "retention": {"name": "d7", "type": "retention", "threshold_days": 7},
    "quantile": {"name": "p50", "type": "quantile", "quantile": 0.5},
    "ratio": {"name": "aov", "type": "ratio", "numerator": "revenue", "denominator": "orders"},
}


def test_synthesise_ratio_metric_applies_window_to_both_measures() -> None:
    metric = synthesise_metric(
        MetricSpec(
            name="aov",
            type="ratio",
            numerator="revenue",
            denominator="orders",
            window_days=14,
        )
    )
    assert isinstance(metric, RatioMetric)
    assert metric.numerator.window_days == 14
    assert metric.denominator.window_days == 14


@pytest.mark.parametrize("metric_type", sorted(_SYNTHESISE_SPEC_KWARGS))
def test_synthesise_metric_leaves_preferred_direction_untracked_when_unset(
    metric_type: str,
) -> None:
    """An undeclared MetricSpec.preferred_direction must not land in the
    synthesised Metric's model_fields_set either, on any of the 5 branches
    -- MetricIdentity's own 'increase' default is a value only, never a
    resolved declaration, so it must stay invisible to the explicitness check
    behind ``Metric.declared_preferred_direction``."""
    spec = MetricSpec(**_SYNTHESISE_SPEC_KWARGS[metric_type])
    metric = synthesise_metric(spec)
    assert "preferred_direction" not in metric.model_fields_set
    assert metric.preferred_direction == "increase"


@pytest.mark.parametrize("metric_type", sorted(_SYNTHESISE_SPEC_KWARGS))
def test_synthesise_metric_forwards_preferred_direction_when_set(metric_type: str) -> None:
    spec = MetricSpec(preferred_direction="decrease", **_SYNTHESISE_SPEC_KWARGS[metric_type])
    metric = synthesise_metric(spec)
    assert "preferred_direction" in metric.model_fields_set
    assert metric.preferred_direction == "decrease"


@pytest.mark.parametrize("metric_type", sorted(_SYNTHESISE_SPEC_KWARGS))
def test_synthesise_metric_explicit_none_preferred_direction_does_not_crash(
    metric_type: str,
) -> None:
    """MetricSpec(preferred_direction=None) is valid, explicit input --
    pydantic tracks an explicit None assignment as "set". It must behave
    identically to leaving the field unset entirely (both mean
    "undeclared"), not get forwarded as a bare None kwarg to the
    synthesised Metric constructor, whose own field has no None in its
    type union (Finding 4: this used to raise a pydantic ValidationError
    naming a type the caller never mentioned)."""
    spec = MetricSpec(preferred_direction=None, **_SYNTHESISE_SPEC_KWARGS[metric_type])
    assert "preferred_direction" in spec.model_fields_set
    metric = synthesise_metric(spec)
    assert "preferred_direction" not in metric.model_fields_set
    assert metric.preferred_direction == "increase"

    omitted = synthesise_metric(MetricSpec(**_SYNTHESISE_SPEC_KWARGS[metric_type]))
    assert metric.model_dump() == omitted.model_dump()


def test_frame_metric_without_declared_direction_preferred_direction_is_none(
    arrow_frame: pa.Table,
) -> None:
    """The actual bug: a frame-path metric that never declares a direction
    must not silently inherit MetricIdentity's "increase" default --
    LiftEstimate.preferred_direction stays None and prob_favorable()
    refuses instead of assuming a direction."""
    analysis = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean")],
        design=_DESIGN,
    )
    (est,) = readouts.run(analysis)
    assert est.preferred_direction is None
    with pytest.raises(InvalidRequestError) as raised:
        est.prob_favorable()
    assert raised.value.code == "estimation.results.lift.liftestimate_prob_favorable"


def test_frame_metric_with_declared_decrease_direction_flips_prob_favorable(
    arrow_frame: pa.Table,
) -> None:
    """MetricSpec(preferred_direction='decrease') (e.g. a latency metric)
    must resolve to a LiftEstimate whose prob_favorable() reads the LOWER
    tail (1 - prob_beyond(null_lift)), not the upper one."""
    analysis = from_unit_summary(
        arrow_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean", preferred_direction="decrease")],
        design=_DESIGN,
    )
    (est,) = readouts.run(analysis)
    assert est.preferred_direction == "decrease"
    assert est.prob_favorable() == pytest.approx(1.0 - est.prob_beyond(est.null_lift))
    assert est.prob_favorable() != pytest.approx(est.prob_beyond(est.null_lift))


def test_metric_spec_accepts_method_roles_and_prior() -> None:
    spec = MetricSpec(
        name="orders",
        covariate="pre_orders",
        decision_method=Method(name="unadjusted"),
        sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
        prior=Normal(mu=0.0, sigma=0.03),
    )
    assert spec.decision_method is not None
    assert spec.sensitivity_methods[0].variance_reduction == "cuped"
    assert spec.prior is not None
    assert spec.prior.sigma == 0.03


def test_metric_spec_method_roles_and_prior_default_to_none() -> None:
    """Omitted method roles/prior inherit call-wide defaults."""
    spec = MetricSpec(name="revenue")
    assert spec.decision_method is None
    assert spec.sensitivity_methods == ()
    assert spec.prior is None


def test_metric_spec_covariate_without_cuped_method_stays_valid() -> None:
    """A covariate declares capability only."""
    spec = MetricSpec(name="revenue", covariate="pre_revenue")
    assert spec.covariate == "pre_revenue"
    assert spec.decision_method is None

    spec_unadjusted_only = MetricSpec(
        name="revenue", covariate="pre_revenue", decision_method=Method(name="unadjusted")
    )
    assert spec_unadjusted_only.decision_method is not None
    assert spec_unadjusted_only.decision_method.variance_reduction == "none"


def test_metric_spec_cuped_method_requires_covariate() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        MetricSpec(
            name="orders",
            decision_method=Method(name="cuped", variance_reduction="cuped"),
        )
    assert exc_info.value.code == "frame.metric.declares_cuped_method"
    assert exc_info.value.context == {"name": "orders"}


def test_metric_spec_cuped_method_accepted_on_ratio_with_a_covariate() -> None:
    """A ratio metric adjusts its numerator and denominator against the one
    declared covariate, so CUPED is a legal declaration for it."""
    spec = MetricSpec(
        name="rpo",
        type="ratio",
        numerator="rev",
        denominator="orders",
        covariate="pre_rpo",
        decision_method=Method(name="cuped", variance_reduction="cuped"),
    )
    assert spec.decision_method is not None
    assert spec.decision_method.variance_reduction == "cuped"
    assert set(spec.source_columns) == {"rev", "pre_rpo", "orders"}


def test_metric_spec_ratio_cuped_still_needs_a_covariate() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        MetricSpec(
            name="rpo",
            type="ratio",
            numerator="rev",
            denominator="orders",
            decision_method=Method(name="cuped", variance_reduction="cuped"),
        )
    assert exc_info.value.code == "frame.metric.declares_cuped_method"


def test_metric_spec_cuped_method_rejected_on_quantile() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        MetricSpec(
            name="latency",
            type="quantile",
            quantile=0.9,
            decision_method=Method(name="cuped", variance_reduction="cuped"),
        )
    assert exc_info.value.code == "frame.metric.cuped_supported_metrics"
    assert exc_info.value.context["name"] == "latency"


def test_frame_source_moments_refuses_unsupported_grain_before_quantile() -> None:
    src = from_unit_summary(
        _arrow_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", type="quantile", quantile=0.9)],
    )
    metric = _metric(src, "revenue")
    with pytest.raises(CapabilityError) as raised:
        src.moments(metric, grain="daily")
    assert raised.value.code == "source.frame.grain"


@pytest.mark.parametrize("grain", ["daily", "asof"])
def test_frame_source_moments_refuses_unsupported_grain(grain: str) -> None:
    src = from_unit_summary(
        _arrow_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
    )
    metric = _metric(src, "revenue")
    with pytest.raises(CapabilityError) as exc_info:
        src.moments(metric, grain=cast(Any, grain))

    assert exc_info.value.code == "source.frame.grain"
    assert exc_info.value.context["grain"] == grain


def test_frame_source_moments_refuses_quantile_metric() -> None:
    src = from_unit_summary(
        _arrow_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", type="quantile", quantile=0.9)],
    )
    metric = _metric(src, "revenue")
    with pytest.raises(CapabilityError) as raised:
        src.moments(metric)
    assert raised.value.code == "source.frame.quantile_no_moments"


def test_quantile_end_to_end_recovers_known_shift() -> None:
    """from_unit_summary -> readouts.run for a quantile metric brackets a
    known +25% shift, mirroring the DGP the estimator unit tests use."""
    rng = np.random.default_rng(11)
    n = 4000
    control = rng.lognormal(1.0, 0.5, n)
    treatment = rng.lognormal(1.0, 0.5, n) * 1.25  # +25% at every quantile
    frame = pa.table(
        {
            "user_id": list(range(2 * n)),
            "variant": ["control"] * n + ["treatment"] * n,
            "latency": np.concatenate([control, treatment]),
        }
    )

    (est,) = readouts.run(
        from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="latency", type="quantile", quantile=0.9)],
            design=_DESIGN,
        )
    )

    assert est.metric == "latency"
    lift = est.require_lift()
    assert 0.15 < lift.value < 0.35
    assert lift.lb is not None and lift.ub is not None
    assert lift.lb < 0.25 < lift.ub


def test_panel_quantile_metric_refuses_asof_moments(panel_frame: pa.Table) -> None:
    src = from_unit_panel(
        panel_frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", type="quantile", quantile=0.9)],
    )

    with pytest.raises(CapabilityError) as exc:
        src.moments(_metric(src, "revenue"), grain="asof")
    assert exc.value.code == "frame.asof.quantile_unsupported"
    assert exc.value.context["metric"] == "revenue"


def test_panel_quantile_metric_resolves_total_moments_under_encouragement_design() -> None:
    """A quantile metric carries no window semantics, so co-registering it
    on an encouragement (uptake) panel must resolve total-grain moments
    cleanly rather than hitting the warehouse-only not-implemented path
    that only conversion/mean metrics used to escape."""
    rows = _pre_exposure_click_rows()
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue", "clicked"],
                cols,
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
        metrics=[MetricSpec(name="revenue", type="quantile", quantile=0.9)],
        uptake="clicked",
        design=_encouragement_design(window_days=2),
        exposure_date="exposed_on",
    )
    metric = _metric(src, "revenue")
    rows_out = src.moments(metric, grain="total")
    assert rows_out


def _encouragement_design(window_days: int | None = None):
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    return Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=window_days),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
    )


def _pre_exposure_click_rows() -> list[tuple[str, str, date, date, float, float]]:
    """u1 (treatment): panel days 0..5, explicit ``exposed_on`` = day 3, its
    only click on day 0 - three days BEFORE its own exposure. u2
    (control): single day-0 row, no clicks."""
    base = date(2025, 1, 1)
    rows = []
    for off in range(6):
        clicked = 1.0 if off == 0 else 0.0
        rows.append(
            ("u1", "treatment", base + timedelta(days=off), base + timedelta(days=3), 1.0, clicked)
        )
    rows.append(("u2", "control", base, base, 1.0, 0.0))
    return rows


def test_panel_asof_winsorization_matches_total_grain() -> None:
    """The declared outcome cap applies to every grain. Cumulated to the last
    observed day the as-of series IS the whole-panel total, so its moments
    must be the winsorized ones grain='total' reports."""
    table = pa.table(
        {
            "user_id": ["c1", "c1", "c2", "c2", "t1", "t1", "t2", "t2"],
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "ds": [date(2026, 1, 1), date(2026, 1, 2)] * 4,
            "revenue": [3.0, 50.0, 1.0, 2.0, 4.0, 60.0, 2.0, 1.0],
        }
    )
    source = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics=[MetricSpec(name="revenue", winsorization={"upper_value": 10.0})],
    )
    metric = source.context.metrics[0]

    total = {row["group_id"]: row for row in source.moments(metric, grain="total")}
    asof = {
        row["group_id"]: row
        for row in source.moments(metric, grain="asof")
        if row["ds"] == date(2026, 1, 2)
    }

    # per-unit totals clipped at 10: control {10, 3} -> 6.5, treatment {10, 3} -> 6.5
    assert total["control"]["ref_y"] == pytest.approx(6.5)
    assert asof["control"]["ref_y"] == pytest.approx(total["control"]["ref_y"])
    assert asof["treatment"]["ref_y"] == pytest.approx(total["treatment"]["ref_y"])
    assert asof["control"]["winsor_upper_bound"] == pytest.approx(10.0)


@pytest.mark.parametrize("grain", ["daily", "asof"])
def test_panel_percentile_winsorization_is_date_local(grain) -> None:
    days = [date(2026, 1, 1), date(2026, 1, 2)]
    table = pa.table(
        {
            "user_id": ["c1", "c1", "c2", "c2", "t1", "t1", "t2", "t2"],
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "ds": days * 4,
            "revenue": [1.0, 100.0, 3.0, 100.0, 5.0, 100.0, 7.0, 100.0],
        }
    )

    def source(data):
        return from_unit_panel(
            data,
            unit="user_id",
            group="variant",
            date="ds",
            control="control",
            metrics=[MetricSpec(name="revenue", winsorization={"upper_percentile": 0.5})],
        )

    prefix = source(table.take(pa.array([0, 2, 4, 6])))
    full = source(table)

    def at(src, day):
        return {
            row["group_id"]: row
            for row in src.moments(src.context.metrics[0], grain=grain)
            if row["ds"] == day
        }

    before, after = at(prefix, days[0]), at(full, days[0])
    assert before["control"]["ref_y"] == pytest.approx(2.0)
    assert before["treatment"]["ref_y"] == pytest.approx(4.0)
    fields = ("n", "ref_y", "cy1", "cy2", "winsor_upper_bound", "winsor_n", "winsor_n_upper")
    for group in ("control", "treatment"):
        assert {k: after[group][k] for k in fields} == pytest.approx(
            {k: before[group][k] for k in fields}
        )
    if grain == "asof":
        total = {r["group_id"]: r for r in full.moments(full.context.metrics[0], grain="total")}
        final = at(full, days[1])
        assert final["treatment"]["winsor_upper_bound"] == pytest.approx(104.0)
        for group in ("control", "treatment"):
            assert {k: final[group][k] for k in fields} == pytest.approx(
                {k: total[group][k] for k in fields}
            )


@pytest.mark.parametrize("grain", ["daily", "asof"])
def test_panel_percentile_winsorization_constant_day_reports_no_cap(grain) -> None:
    """A densified/all-zero day resolves lower == upper from a percentile
    cutoff - clipping a constant column to itself is a no-op, so that date
    must report the unconfigured "no cap" representation rather than a
    degenerate lower/upper pair."""
    days = [date(2026, 1, 1), date(2026, 1, 2)]
    table = pa.table(
        {
            "user_id": ["c1", "c2", "t1", "t2"] * 2,
            "variant": ["control", "control", "treatment", "treatment"] * 2,
            "ds": [days[0]] * 4 + [days[1]] * 4,
            "revenue": [0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 5.0, 50.0],
        }
    )

    def source(winsorized: bool):
        metrics = (
            [
                MetricSpec(
                    name="revenue", winsorization={"lower_percentile": 0.1, "upper_percentile": 0.9}
                )
            ]
            if winsorized
            else [MetricSpec(name="revenue")]
        )
        return from_unit_panel(
            table, unit="user_id", group="variant", date="ds", control="control", metrics=metrics
        )

    capped, uncapped = source(True), source(False)

    def rows_on(src, day):
        return [r for r in src.moments(src.context.metrics[0], grain=grain) if r["ds"] == day]

    constant_rows = rows_on(capped, days[0])
    varied_rows = rows_on(capped, days[1])
    fields = ("n", "ref_y", "cy1", "cy2")
    unclipped_by_group = {r["group_id"]: r for r in rows_on(uncapped, days[0])}
    for row in constant_rows:
        assert row["winsor_lower_bound"] is None
        assert row["winsor_upper_bound"] is None
        assert row["winsor_n"] is None
        assert {k: row[k] for k in fields} == pytest.approx(
            {k: unclipped_by_group[row["group_id"]][k] for k in fields}
        )
    for row in varied_rows:
        assert row["winsor_lower_bound"] is not None
        assert row["winsor_upper_bound"] is not None
        assert row["winsor_lower_bound"] < row["winsor_upper_bound"]


@pytest.mark.parametrize("grain", ["daily", "asof"])
def test_panel_percentile_winsorization_refuses_collapsed_bound_that_clips(grain) -> None:
    """19 zeros and one 500 resolve lower == upper == 0.0 from a 10th/90th
    percentile cutoff - unlike the all-zero no-op case, clipping to that
    collapsed pair drags the 500 down to 0 while the row would report no
    cap at all. Refuse rather than silently return the wrong moment."""
    day = date(2026, 1, 1)
    users = [f"u{i}" for i in range(20)]
    table = pa.table(
        {
            "user_id": users,
            "variant": ["control"] * 10 + ["treatment"] * 10,
            "ds": [day] * 20,
            "revenue": [0.0] * 19 + [500.0],
        }
    )
    source = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue", winsorization={"lower_percentile": 0.1, "upper_percentile": 0.9}
            )
        ],
    )
    metric = source.context.metrics[0]
    with pytest.raises(CapabilityError) as exc_info:
        source.moments(metric, grain=grain)
    assert exc_info.value.code == "frame.winsorization.collapsed_bound"
    assert exc_info.value.context["metric"] == "revenue"
    assert exc_info.value.context["date"] == day
    assert exc_info.value.context["lower"] == pytest.approx(0.0)
    assert exc_info.value.context["upper"] == pytest.approx(0.0)
    assert exc_info.value.context["clipped"] == 1


@pytest.mark.parametrize("grain", ["daily", "asof"])
def test_panel_percentile_winsorization_collapsed_bound_isolated_to_its_metric(grain) -> None:
    """A collapsed-bound refusal on one metric must not block a co-registered
    valid metric sharing the same batched daily/as-of read, and the refusal
    must still fire on a later direct request for the bad metric even after
    the good metric was served from the same cached batch."""
    day = date(2026, 1, 1)
    users = [f"u{i}" for i in range(20)]
    table = pa.table(
        {
            "user_id": users,
            "variant": ["control"] * 10 + ["treatment"] * 10,
            "ds": [day] * 20,
            "revenue": [0.0] * 19 + [500.0],
            "orders": [1.0] * 20,
        }
    )
    source = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue", winsorization={"lower_percentile": 0.1, "upper_percentile": 0.9}
            ),
            MetricSpec(name="orders"),
        ],
    )
    bad, good = _metric(source, "revenue"), _metric(source, "orders")

    good_rows = source.moments(good, grain=grain)
    assert good_rows

    with pytest.raises(CapabilityError) as exc_info:
        source.moments(bad, grain=grain)
    assert exc_info.value.code == "frame.winsorization.collapsed_bound"

    # The good metric's own cached batch survives the sibling's refusal ...
    assert source.moments(good, grain=grain) == good_rows
    # ... and the bad metric's refusal is cached too, not dropped.
    with pytest.raises(CapabilityError) as exc_info_again:
        source.moments(bad, grain=grain)
    assert exc_info_again.value.code == "frame.winsorization.collapsed_bound"


def _totals_source(metrics, **kwargs):
    return from_unit_summary(
        _arrow_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=metrics,
        **kwargs,
    )


def _frame_quantile_no_moments():
    src = _totals_source([MetricSpec(name="revenue", type="quantile", quantile=0.9)])
    src.moments(_metric(src, "revenue"))


def _frame_breakouts_unsupported():
    src = _totals_source({"revenue": "mean"})
    src.moments(_metric(src, "revenue"), by=["country"])


def _frame_sql_unsupported():
    _totals_source({"revenue": "mean"}).sql()


def _panel_breakout_table() -> pa.Table:
    rows = [
        ("u1", "control", date(2025, 1, 1), "US", "gold", 4.0),
        ("u1", "control", date(2025, 1, 2), "US", "gold", 8.0),
        ("u2", "treatment", date(2025, 1, 1), "CA", "silver", 3.0),
        ("u2", "treatment", date(2025, 1, 2), "CA", "silver", 6.0),
    ]
    cols = list(zip(*rows, strict=True))
    return pa.table(
        dict(zip(["user_id", "variant", "day", "country", "plan", "revenue"], cols, strict=True))
    )


def _frame_multi_dimension():
    src = from_unit_panel(
        _panel_breakout_table(),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country", "plan"],
    )
    src.moments(_metric(src, "revenue"), by=["country", "plan"])


def _frame_undeclared_breakout():
    src = from_unit_panel(
        _panel_breakout_table(),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country"],
    )
    src.moments(_metric(src, "revenue"), by=["plan"])


def _frame_windowed_encouragement_total():
    rows = _pre_exposure_click_rows()
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(
            zip(
                ["user_id", "variant", "day", "exposed_on", "revenue", "clicked"], cols, strict=True
            )
        )
    )
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=2)],
        uptake="clicked",
        design=_encouragement_design(window_days=2),
        exposure_date="exposed_on",
    )
    src.moments(_metric(src, "revenue"))


def _frame_retention_daily():
    rows = [
        ("u1", "control", date(2025, 1, 1), date(2025, 1, 1), 1.0),
        ("u1", "control", date(2025, 1, 2), date(2025, 1, 1), 0.0),
        ("u2", "treatment", date(2025, 1, 1), date(2025, 1, 1), 1.0),
        ("u2", "treatment", date(2025, 1, 2), date(2025, 1, 1), 1.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(["user_id", "variant", "day", "exposed_on", "m"], cols, strict=True)))
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="d7", type="retention", value_column="m", threshold_days=1)],
        exposure_date="exposed_on",
    )
    src.moments(_metric(src, "d7"), grain="daily")


def _frame_quantile_daily():
    src = from_unit_panel(
        _panel_table(),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", type="quantile", quantile=0.9)],
    )
    src.moments(_metric(src, "revenue"), grain="daily")


def _frame_unit_frame_panel():
    """`unit_frame` still refuses on a panel for a retention metric --
    its per-unit value depends on evaluating the retention band against
    the full day axis, which this per-unit sum does not cover. A
    quantile metric can never declare `window_days` on the frame path
    (`frame.metric.window_days_supported`), so unwindowed quantile is
    served the same way mean/ratio/conversion already are (see
    tests/test_frame_panel_unit_frame.py)."""
    rows = [
        ("u1", "control", date(2025, 1, 1), date(2025, 1, 1), 1.0),
        ("u2", "treatment", date(2025, 1, 1), date(2025, 1, 1), 0.0),
        ("u2", "treatment", date(2025, 1, 2), date(2025, 1, 1), 1.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(["user_id", "variant", "day", "exposed_on", "m"], cols, strict=True)))
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="d7", type="retention", value_column="m", threshold_days=1)],
        exposure_date="exposed_on",
    )
    src.unit_frame(_metric(src, "d7"))


def _frame_panel_cluster_arg():
    from_unit_panel(
        _panel_table(),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        cluster="user_id",
    )


_FRAME_MESSAGE_CASES = [
    ("source.frame.quantile_no_moments", _frame_quantile_no_moments),
    ("source.frame.breakouts_unsupported", _frame_breakouts_unsupported),
    ("source.frame.sql_unsupported", _frame_sql_unsupported),
    ("source.frame.multi_dimension", _frame_multi_dimension),
    ("source.frame.undeclared_breakout", _frame_undeclared_breakout),
    ("source.frame.windowed_encouragement_total", _frame_windowed_encouragement_total),
    ("source.frame.retention_daily", _frame_retention_daily),
    ("source.frame.quantile_daily", _frame_quantile_daily),
    ("source.frame.unit_frame_panel", _frame_unit_frame_panel),
]


@pytest.mark.parametrize(("expected_code", "builder"), _FRAME_MESSAGE_CASES)
def test_frame_message_call_sites_are_individually_coded(expected_code, builder) -> None:
    """Each former ``_FRAME_MESSAGE`` guard now carries its own code, so a
    caller can branch on the specific capability gap without parsing text."""
    with pytest.raises(CapabilityError) as raised:
        builder()
    assert raised.value.code == expected_code


def test_from_unit_panel_cluster_shares_the_cluster_grain_code() -> None:
    """from_unit_panel(cluster=...) reports under the same code as the
    sibling cluster_counts() capability gap, naming its own operation."""
    with pytest.raises(CapabilityError) as raised:
        _frame_panel_cluster_arg()
    assert raised.value.code == "source.frame_panel.cluster_grain"
    assert raised.value.context["operation"] == "from_unit_panel(cluster=...)"
    assert raised.value.context["cluster"] == "user_id"


def _missing_metric_frame():
    frame = pa.table({"unit": ["a", "b"], "group": ["control", "treatment"], "y": [1.0, 2.0]})
    from_unit_summary(
        frame, unit="unit", group="group", control="control", metrics={"missing": "mean"}
    )


def _duplicate_unit_frame():
    frame = pa.table({"unit": ["a", "a"], "group": ["control", "treatment"], "y": [1.0, 2.0]})
    from_unit_summary(frame, unit="unit", group="group", control="control", metrics={"y": "mean"})


def _missing_control_frame():
    frame = pa.table({"unit": ["a", "b"], "group": ["x", "y"], "y": [1.0, 2.0]})
    from_unit_summary(frame, unit="unit", group="group", control="control", metrics={"y": "mean"})


def _uptake_not_binary_panel():
    rows = [
        ("u1", "control", date(2025, 1, 1), 0.0, 4.0),
        ("u1", "control", date(2025, 1, 2), 2.0, 4.0),
        ("u2", "treatment", date(2025, 1, 1), 1.0, 4.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "clicked", "revenue"], cols, strict=True))
    )
    from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
    )


def _conversion_not_binary_frame():
    frame = pa.table(
        {
            "unit": ["a", "b", "c"],
            "group": ["control", "control", "treatment"],
            "converted": [0.0, 2.0, 1.0],
        }
    )
    from_unit_summary(
        frame,
        unit="unit",
        group="group",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion")],
    )


def _cluster_labels_frame():
    frame = pa.table(
        {
            "unit": ["a", "b", "c"],
            "group": ["control", "control", "treatment"],
            "cluster": ["s1", None, "s2"],
            "y": [1.0, 2.0, 3.0],
        }
    )
    from_unit_summary(
        frame,
        unit="unit",
        group="group",
        control="control",
        metrics={"y": "mean"},
        cluster="cluster",
    )


def _day_grain_columns_panel():
    table = pa.table(
        {
            "user_id": ["a", "b"],
            "variant": ["control", "treatment"],
            "day": [
                datetime(2025, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 1, tzinfo=UTC),
            ],
            "revenue": [1.0, 2.0],
        }
    )
    from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )


def _null_identity_frame():
    frame = pa.table({"unit": [None, "b"], "group": ["control", "treatment"], "y": [1.0, 2.0]})
    from_unit_summary(frame, unit="unit", group="group", control="control", metrics={"y": "mean"})


def _non_finite_frame():
    frame = pa.table(
        {"unit": ["a", "b"], "group": ["control", "treatment"], "y": [1.0, float("inf")]}
    )
    from_unit_summary(frame, unit="unit", group="group", control="control", metrics={"y": "mean"})


def _unassigned_frame():
    frame = pa.table(
        {"unit": ["a", "b", "c"], "group": ["control", None, "treatment"], "y": [1.0, 2.0, 3.0]}
    )
    from_unit_summary(frame, unit="unit", group="group", control="control", metrics={"y": "mean"})


def _null_exposure_panel():
    rows = [
        ("u1", "control", date(2025, 1, 1), None, 1.0),
        ("u2", "treatment", date(2025, 1, 1), date(2025, 1, 1), 2.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(
        dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], cols, strict=True))
    )
    from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        exposure_date="exposed_on",
    )


def _exposure_date_required_panel():
    from_unit_panel(
        _panel_table(),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="revenue", window_days=2)],
    )


def _duplicate_unit_days_panel():
    rows = [
        ("u1", "control", date(2025, 1, 1), 1.0),
        ("u1", "control", date(2025, 1, 1), 2.0),
        ("u2", "treatment", date(2025, 1, 1), 3.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(["user_id", "variant", "day", "revenue"], cols, strict=True)))
    from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )


def _multi_group_unit_panel():
    rows = [
        ("u1", "control", date(2025, 1, 1), 1.0),
        ("u1", "treatment", date(2025, 1, 2), 2.0),
        ("u2", "treatment", date(2025, 1, 1), 3.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(["user_id", "variant", "day", "revenue"], cols, strict=True)))
    from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )


def _ratio_same_column_frame():
    frame = pa.table({"unit": ["a", "b"], "group": ["control", "treatment"], "y": [1.0, 2.0]})
    from_unit_summary(
        frame,
        unit="unit",
        group="group",
        control="control",
        metrics=[MetricSpec(name="r", type="ratio", numerator="y", denominator="y")],
    )


def _on_unassigned_invalid_frame():
    frame = pa.table({"unit": ["a", "b"], "group": ["control", "treatment"], "y": [1.0, 2.0]})
    from_unit_summary(
        frame,
        unit="unit",
        group="group",
        control="control",
        metrics={"y": "mean"},
        on_unassigned=cast(Any, "bogus"),
    )


_FRAME_VALIDATION_CASES = [
    ("source.frame.metric_missing", _missing_metric_frame),
    ("source.frame.duplicate_units", _duplicate_unit_frame),
    ("source.frame.control_missing", _missing_control_frame),
    ("source.frame.uptake_not_binary", _uptake_not_binary_panel),
    ("source.frame.conversion_not_binary", _conversion_not_binary_frame),
    ("source.frame.cluster_labels", _cluster_labels_frame),
    ("source.panel.day_grain_columns", _day_grain_columns_panel),
    ("source.frame.null_identity", _null_identity_frame),
    ("source.frame.non_finite", _non_finite_frame),
    ("source.frame.unassigned", _unassigned_frame),
    ("source.panel.null_exposure", _null_exposure_panel),
    ("source.panel.exposure_date", _exposure_date_required_panel),
    ("source.panel.duplicate_unit_days", _duplicate_unit_days_panel),
    ("source.panel.multi_group_unit", _multi_group_unit_panel),
    ("source.frame.ratio_same_column", _ratio_same_column_frame),
    ("source.frame.on_unassigned_invalid", _on_unassigned_invalid_frame),
]


@pytest.mark.parametrize(("expected_code", "builder"), _FRAME_VALIDATION_CASES)
def test_frame_validation_refusals_are_coded(expected_code, builder) -> None:
    with pytest.raises(InvalidRequestError) as raised:
        builder()
    assert raised.value.code == expected_code
    if expected_code == "source.frame.metric_missing":
        assert raised.value.context["available"] == tuple(sorted(["unit", "group", "y"]))
    elif expected_code == "source.frame.ratio_same_column":
        assert raised.value.context["metric"] == "r"
        assert raised.value.context["column"] == "y"
    elif expected_code == "source.frame.on_unassigned_invalid":
        assert raised.value.context["value"] == "bogus"
        assert raised.value.context["allowed"] == {"error", "exclude"}
    elif expected_code == "source.frame.cluster_labels":
        assert raised.value.context["reason"] == "null_label"
        assert raised.value.context["cluster"] == "cluster"
        assert raised.value.context["missing"] == 1
    elif expected_code in {
        "source.frame.uptake_not_binary",
        "source.frame.conversion_not_binary",
    }:
        assert raised.value.context["values"] == (2.0,)
    restored = pickle.loads(pickle.dumps(raised.value))
    assert restored.code == raised.value.code
    assert restored.context == raised.value.context


def _type_but_sets_metric():
    MetricSpec(name="revenue", type="mean", numerator="a")


def _sets_covariate_missing_metric():
    MetricSpec(name="revenue", covariate_missing="zero")


def _metrics_entries_not_metricspec():
    coerce_metrics([123])  # ty: ignore[invalid-argument-type]


def _duplicate_metric_name_specs():
    coerce_metrics([MetricSpec(name="a"), MetricSpec(name="a")])


def _terse_retention_type():
    coerce_metrics({"revenue": "retention"})


def _terse_quantile_type():
    coerce_metrics({"revenue": "quantile"})


def _terse_unknown_type():
    coerce_metrics({"revenue": "bogus"})


def _frame_asof_completed_windows_contradictory():
    rows = [
        ("u1", "control", date(2025, 1, 1), date(2025, 1, 1), 1.0),
        ("u1", "control", date(2025, 1, 2), date(2025, 1, 1), 0.0),
        ("u2", "treatment", date(2025, 1, 1), date(2025, 1, 1), 1.0),
        ("u2", "treatment", date(2025, 1, 2), date(2025, 1, 1), 1.0),
    ]
    cols = list(zip(*rows, strict=True))
    table = pa.table(dict(zip(["user_id", "variant", "day", "exposed_on", "m"], cols, strict=True)))
    src = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[MetricSpec(name="d7", type="retention", value_column="m", threshold_days=1)],
        exposure_date="exposed_on",
    )
    src.moments(_metric(src, "d7"), grain="asof", completed_windows_only=True)


_FRAME_SPEC_REFUSAL_CASES = [
    ("frame.metric.type_but_sets", _type_but_sets_metric, {"name": "revenue", "type": "mean"}),
    (
        "frame.metric.sets_covariate_missing",
        _sets_covariate_missing_metric,
        {"covariate_missing": "zero", "name": "revenue"},
    ),
    ("frame.metrics_entries_metricspec", _metrics_entries_not_metricspec, {"type_name": "int"}),
    ("frame.duplicate_metric_name", _duplicate_metric_name_specs, {"name": "a"}),
    ("frame.metric_type_retention", _terse_retention_type, {"name": "revenue"}),
    ("frame.metric_type_quantile", _terse_quantile_type, {"name": "revenue"}),
    ("frame.metric_unknown_type", _terse_unknown_type, {"kind": "bogus", "name": "revenue"}),
    (
        "frame.frame_panel.asof_moments_completed",
        _frame_asof_completed_windows_contradictory,
        {"band": (1, None), "name": "d7"},
    ),
]


@pytest.mark.parametrize(
    ("expected_code", "builder", "expected_context"), _FRAME_SPEC_REFUSAL_CASES
)
def test_frame_spec_refusals_are_coded(expected_code, builder, expected_context) -> None:
    """Each converted bare raise in MetricSpec/coerce_metrics/_as_type/
    FramePanelSource carries its own stable code and full context."""
    with pytest.raises(InvalidRequestError) as raised:
        builder()
    assert raised.value.code == expected_code
    assert raised.value.context == expected_context


def test_frame_reexports_metric_specs_identity() -> None:
    assert frame.MetricSpec is metric_specs.MetricSpec
    assert frame.coerce_metrics is metric_specs.coerce_metrics
    assert frame.synthesise_metric is metric_specs.synthesise_metric


def test_metric_spec_impute_missing_is_a_named_refusal_with_a_route() -> None:
    with pytest.raises(InvalidRequestError) as raised:
        MetricSpec(name="revenue", missing="impute")
    assert raised.value.code == "frame.metric.missing_impute"
    assert raised.value.context["metric"] == "revenue"
    assert "route" in raised.value.context
