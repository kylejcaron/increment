from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import pyarrow as pa
import pytest

from increment import readouts
from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
)
from increment.estimation.engine import Method
from increment.frame import MetricSpec, from_unit_panel, from_unit_summary
from increment.semantics.design import Randomized
from increment.sources import Grain, MomentSource, SourceOperation
from tests.sequential_cases import registration


def test_component_identity_includes_optional_source_and_dimension_coordinates():
    from increment.readouts._source_digest import component, composite_source

    digest = "a" * 64
    first = component(
        kind="moments_rows",
        metric="revenue",
        population="assigned",
        sha256=digest,
        source="warehouse",
        dimension="country",
    )
    same = component(
        kind="moments_rows",
        metric="revenue",
        population="assigned",
        sha256=digest,
        source="warehouse",
        dimension="country",
    )
    other_dimension = component(
        kind="moments_rows",
        metric="revenue",
        population="assigned",
        sha256=digest,
        source="warehouse",
        dimension="device",
    )

    assert first == same
    assert first != other_dimension
    assert first["source"] == "warehouse"
    assert first["dimension"] == "country"
    assert (
        component(
            kind="assignment_counts",
            metric=None,
            population="assigned",
            sha256=digest,
        )["source"]
        is None
    )
    assert composite_source([first]) == composite_source([same])
    assert composite_source([first]) != composite_source([other_dimension])
    from increment.readouts._randomized_scope import _randomized_snapshot_id

    request = {"view": "breakout"}
    first_id = _randomized_snapshot_id(composite_source([first]), request)
    same_id = _randomized_snapshot_id(composite_source([same]), request)
    other_dimension_id = _randomized_snapshot_id(composite_source([other_dimension]), request)
    assert first_id == same_id
    assert first_id != other_dimension_id
    from types import SimpleNamespace

    from increment.readouts._randomized_scope import _randomized_source

    selected = [SimpleNamespace(name="revenue")]
    evidence = {"revenue": ("moments_rows", digest, 1)}
    partition_source = _randomized_source(
        selected,
        evidence,
        None,
        "assigned",
        source="warehouse",
        dimension="country",
    )
    repeated_source = _randomized_source(
        selected,
        evidence,
        None,
        "assigned",
        source="warehouse",
        dimension="country",
    )
    other_source = _randomized_source(
        selected,
        evidence,
        None,
        "assigned",
        source="warehouse",
        dimension="device",
    )
    assert _randomized_snapshot_id(partition_source, request) == _randomized_snapshot_id(
        repeated_source, request
    )
    assert _randomized_snapshot_id(partition_source, request) != _randomized_snapshot_id(
        other_source, request
    )
    assert composite_source([first, same])["components"] == [first, same]
    import json

    restored_source = json.loads(json.dumps(partition_source))
    assert restored_source == partition_source
    assert _randomized_snapshot_id(restored_source, request) == _randomized_snapshot_id(
        partition_source, request
    )


class _CountingSource:
    operations: frozenset[SourceOperation] = frozenset()

    def __init__(self, source):
        self._source = source
        self.moment_calls = 0

    @property
    def context(self):
        return self._source.context

    @property
    def capabilities(self) -> frozenset[Grain]:
        return self._source.capabilities

    @property
    def breakouts(self):
        return self._source.breakouts

    @property
    def shape(self):
        return self._source.shape

    def moments(self, *args, **kwargs):
        self.moment_calls += 1
        raise AssertionError("readout queried moments before validating the request")


def _panel_source(*, breakouts=("segment",), design=None):
    table = pa.table(
        {
            "unit": [f"u{i}" for i in range(8)],
            "variant": ["control", "treatment"] * 4,
            "day": np.tile(np.array([1, 2], dtype="int64"), 4),
            "segment": ["a", "a", "b", "b", "a", "a", "b", "b"],
            "metric": np.ones(8),
        }
    )
    return from_unit_panel(
        table,
        unit="unit",
        group="variant",
        date="day",
        metrics=[MetricSpec(name="metric")],
        control="control",
        breakouts=list(breakouts),
        design=design,
    )


def test_readout_request_validates_source_dimension_before_query():
    source = _CountingSource(_panel_source())

    with pytest.raises(CapabilityError) as raised:
        readouts.run(cast(MomentSource, source), by=["not_declared"])

    assert getattr(raised.value, "code", None) == "readout.source.dimension"
    assert source.moment_calls == 0


def test_readout_request_rejects_dimensions_when_source_declares_none():
    source = _CountingSource(_panel_source(breakouts=()))

    with pytest.raises(CapabilityError) as raised:
        readouts.run(cast(MomentSource, source), by=["segment"])

    assert getattr(raised.value, "code", None) == "readout.source.dimension"
    assert source.moment_calls == 0


def test_run_refuses_nonempty_by_instead_of_dropping_segments():
    """Reject breakouts rather than merge segments under (metric, group_id).

    Declaring a dimension on the source does not make run() segment-aware.
    """
    source = _panel_source(breakouts=("segment",))

    with pytest.raises(UnsupportedRequestError) as raised:
        readouts.run(cast(MomentSource, source), by=["segment"])

    assert getattr(raised.value, "code", None) == "readout.run.segment_unsupported"


def test_arm_cluster_sequential_refusal_uses_the_canonical_compatibility_code() -> None:
    from increment.compatibility import ARM_COMPATIBILITY_REFUSALS

    spec = ARM_COMPATIBILITY_REFUSALS["arm.inference.cluster"]
    assert isinstance(spec, RefusalSpec)
    assert spec.code == "arm.inference.cluster"
    assert spec.error_type is CapabilityError


def test_sequential_run_refuses_plan_q_mismatch_before_read_and_preserves_refusal():
    import copy
    import pickle
    from fractions import Fraction

    from increment._analysis_config import resolve_configs
    from increment.frame import synthesise_metric
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan, InferenceSpec
    from increment.sources import SourceContext

    reg = registration("bernoulli").model_copy(update={"q": Fraction(1, 5)})
    metric = synthesise_metric(MetricSpec(name="outcome", type="conversion"))
    design = Randomized(control_group="control")
    declaration = AnalysisPlan(
        primary="outcome",
        q=0.10,
        inference=InferenceSpec(kind="always_valid", registration=reg),
    )
    configs = resolve_configs((metric,), None, None, methods=None, prior=None)
    plan = compile_decision_plan(
        declaration, (metric,), path="frame", design=design, configs=configs
    )
    context = SourceContext(
        study_id=reg.source_id,
        design=design,
        plan=plan,
        metrics=(metric,),
        configs=configs,
        cluster=None,
    )

    class UnreadSource:
        capabilities = frozenset({"total"})
        breakouts = ()
        shape = None

        def __init__(self):
            self.context = context
            self.moment_calls = 0

        def moments(self, *args, **kwargs):
            self.moment_calls += 1
            raise AssertionError("sequential q refusal must precede source reads")

    source = UnreadSource()
    with pytest.raises(CapabilityError) as raised:
        readouts.run(cast(MomentSource, source))
    refusal = raised.value
    assert refusal.code == "sequential.source.invalid"
    assert set(refusal.context) == {"reason"}
    assert isinstance(refusal.context["reason"], str)
    assert source.moment_calls == 0
    for restored in (copy.deepcopy(refusal), pickle.loads(pickle.dumps(refusal))):
        assert restored.code == refusal.code
        assert restored.context == refusal.context


def test_unsupported_readout_refusal_keeps_not_implemented_catch() -> None:
    """`arm.metric.quantile_cuped` is the canonical compatibility code for
    the quantile+CUPED hazard, raised via `refuse_unsupported` from every
    entry point, not a per-module duplicate."""
    from increment.compatibility import ARM_COMPATIBILITY_REFUSALS, Unsupported, refuse_unsupported

    spec = ARM_COMPATIBILITY_REFUSALS["arm.metric.quantile_cuped"]
    assert spec.error_type is UnsupportedRequestError

    with pytest.raises(NotImplementedError) as raised:
        refuse_unsupported(Unsupported("arm.metric.quantile_cuped"), metric="q")
    assert isinstance(raised.value, UnsupportedRequestError)
    assert raised.value.code == "arm.metric.quantile_cuped"


def test_encouragement_breakout_sequential_refuses_with_stable_code() -> None:
    from types import SimpleNamespace

    from increment.estimation.encouragement import validate_readout_encouragement
    from increment.estimation.sequential import AlwaysValid

    request = SimpleNamespace(
        design=SimpleNamespace(mechanism="encouragement"),
        metrics=(),
        configs=(),
        estimands=None,
        plan=SimpleNamespace(inference=AlwaysValid(registration=registration("gaussian"))),
        cluster=None,
        value_scale=None,
        view="breakout",
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        validate_readout_encouragement(request)  # ty: ignore[invalid-argument-type] -- minimal stand-in for ReadoutRequest
    assert raised.value.code == "readout.inference.sequential_view"


def test_encouragement_value_scale_refuses_with_stable_code() -> None:
    from types import SimpleNamespace

    from increment.estimation.encouragement import validate_readout_encouragement

    request = SimpleNamespace(
        design=SimpleNamespace(mechanism="encouragement"),
        metrics=(),
        configs=(),
        estimands=None,
        plan=SimpleNamespace(inference=None),
        cluster=None,
        value_scale={"rev": "absolute"},
        view="run",
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        validate_readout_encouragement(request)  # ty: ignore[invalid-argument-type] -- minimal stand-in for ReadoutRequest
    assert raised.value.code == "readout.encouragement.value_scale"


def test_encouragement_clustered_ratio_metric_refuses_with_stable_code() -> None:
    from types import SimpleNamespace

    from increment.estimation.encouragement import validate_readout_encouragement

    ratio_metric = SimpleNamespace(name="conv_rate", type="ratio", margin=None, margin_abs=None)
    request = SimpleNamespace(
        design=SimpleNamespace(mechanism="encouragement"),
        metrics=(ratio_metric,),
        configs=(),
        estimands=None,
        plan=SimpleNamespace(
            inference=None,
            procedures={"conv_rate": SimpleNamespace(null_lift=0.0, null_abs=None)},
        ),
        cluster="store",
        value_scale=None,
        view="run",
    )
    with pytest.raises(CapabilityError) as raised:
        validate_readout_encouragement(request)  # ty: ignore[invalid-argument-type] -- minimal stand-in for ReadoutRequest
    assert raised.value.code == "readout.encouragement.cluster_ratio"
    assert raised.value.context["cluster"] == "store"
    assert "conv_rate" in cast(tuple[str, ...], raised.value.context["names"])


def test_encouragement_retention_metric_refuses_with_stable_code() -> None:
    from types import SimpleNamespace

    from increment.estimation.encouragement import validate_readout_encouragement

    retention_metric = SimpleNamespace(name="ret", type="retention", margin=None, margin_abs=None)
    request = SimpleNamespace(
        design=SimpleNamespace(mechanism="encouragement"),
        metrics=(retention_metric,),
        configs=(),
        estimands=None,
        plan=SimpleNamespace(
            inference=None,
            procedures={"ret": SimpleNamespace(null_lift=0.0, null_abs=None)},
        ),
        cluster=None,
        value_scale=None,
        view="run",
    )
    with pytest.raises(CapabilityError) as raised:
        validate_readout_encouragement(request)  # ty: ignore[invalid-argument-type] -- minimal stand-in for ReadoutRequest
    assert raised.value.code == "readout.encouragement.retention"
    assert raised.value.context == {"names": ("ret",)}


def test_encouragement_margin_without_itt_refuses_with_stable_code() -> None:
    from types import SimpleNamespace

    from increment.estimation.encouragement import validate_readout_encouragement

    margin_metric = SimpleNamespace(name="rev", type="mean", margin=0.05, margin_abs=None)
    request = SimpleNamespace(
        design=SimpleNamespace(mechanism="encouragement"),
        metrics=(margin_metric,),
        configs=(),
        estimands=("late",),
        plan=SimpleNamespace(
            inference=None,
            procedures={"rev": SimpleNamespace(null_lift=0.0, null_abs=None)},
        ),
        cluster=None,
        value_scale=None,
        view="run",
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        validate_readout_encouragement(request)  # ty: ignore[invalid-argument-type] -- minimal stand-in for ReadoutRequest
    assert raised.value.code == "readout.encouragement.margin"
    assert "rev" in cast(tuple[str, ...], raised.value.context["names"])


def test_breakout_request_rejects_an_unserved_dimension_before_query():
    source = _CountingSource(
        _panel_source(
            breakouts=(),
            design=Randomized(control_group="control"),
        )
    )

    with pytest.raises(CapabilityError) as raised:
        readouts.breakout(cast(MomentSource, source), "segment", metrics=["metric"])

    assert getattr(raised.value, "code", None) == "readout.source.dimension"
    assert source.moment_calls == 0


def test_breakout_request_validates_correction_before_query():
    source = _CountingSource(
        _panel_source(
            breakouts=("segment",),
            design=Randomized(control_group="control"),
        )
    )

    with pytest.raises(InvalidRequestError) as raised:
        readouts.breakout(
            cast(MomentSource, source),
            "segment",
            metrics=["metric"],
            correction="invalid",  # ty: ignore[invalid-argument-type]
            q=0.03,
        )

    assert raised.value.code == "readout.correction.invalid"
    assert source.moment_calls == 0


def test_breakout_request_accepts_a_served_dimension():
    source = _panel_source(
        breakouts=("segment",),
        design=Randomized(control_group="control"),
    )

    with warnings.catch_warnings():
        # Arms with identical outcomes are reported as excluded rows.
        warnings.simplefilter("ignore")
        estimates = readouts.breakout(cast(MomentSource, source), "segment", metrics=["metric"])

    assert {row.dimension_value for row in estimates} == {"a", "b"}
    assert {row.metric for row in estimates} == {"metric"}


def test_observational_segmented_run_uses_stable_refusal_before_source_dimension():
    from increment.semantics.design import AdjustmentSet, Observational

    for breakouts in ((), ("segment",)):
        source = _CountingSource(
            _panel_source(
                breakouts=breakouts,
                design=Observational(
                    control_group="control",
                    adjustment=AdjustmentSet(covariates=("metric",)),
                ),
            )
        )

        with pytest.raises(NotImplementedError) as raised:
            readouts.run(cast(MomentSource, source), by=["segment"])

        assert getattr(raised.value, "code", None) == "readout.view.observational"
        assert source.moment_calls == 0


def test_sequential_cuped_refuses_before_frame_access():
    from increment.errors import CodedError
    from increment.frame import synthesise_metric
    from tests.sequential_cases import declared_plan

    specs = [
        MetricSpec(
            name="metric",
            covariate="covariate",
            decision_method=Method(name="cuped", variance_reduction="cuped"),
        )
    ]
    design = Randomized(control_group="control")
    plan = declared_plan(
        [synthesise_metric(spec) for spec in specs],
        source_id="frame",
        design=design,
        transformations=specs,
    )

    from tests.sequential_cases import UnreadFrame

    with pytest.raises(CodedError) as raised:
        from_unit_summary(
            UnreadFrame(),
            unit="unit",
            group="variant",
            control="control",
            metrics=specs,
            design=design,
            plan=plan,
        )
    assert raised.value.code == "plan.metric_cuped_methods"


def test_sequential_unproved_sensitivity_refuses_before_frame_access():
    from increment.errors import CapabilityError
    from tests.test_sequential_public_sources import gaussian_plan

    specs = [
        MetricSpec(
            name="metric", type="conversion", sensitivity_methods=(Method(name="sensitivity"),)
        )
    ]
    plan = gaussian_plan(specs, law="bernoulli")

    from tests.sequential_cases import UnreadFrame

    with pytest.raises(CapabilityError) as raised:
        from_unit_summary(
            UnreadFrame(),
            unit="unit",
            group="variant",
            control="control",
            metrics=specs,
            design=Randomized(control_group="control"),
            plan=plan,
        )
    assert raised.value.code == "sequential.route.unsupported"
