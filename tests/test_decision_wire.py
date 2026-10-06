"""Focused contract tests for the compiled decision-plan wire format."""

from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from pydantic import BaseModel, ValidationError

from increment.decision import (
    AbsoluteArmDecisionProcedure,
    CompiledDecisionPlan,
    CompiledViewPolicies,
    ContrastDecisionProcedure,
    FamilyMembership,
    FixedInference,
    MultiplicityFamily,
    NoFamily,
    RelativeArmDecisionProcedure,
)
from increment.decision_wire import (
    WireAbsoluteArmProcedure,
    WireAlwaysValid,
    WireCompiledDecisionPlan,
    WireContrastProcedure,
    WireFamilyMembership,
    WireFixedInference,
    WireMethod,
    WireMultiplicityFamily,
    WireNoFamily,
    WireRelativeArmProcedure,
    WireViewPolicies,
    compiled_plan_from_dict,
    compiled_plan_from_dto,
    compiled_plan_from_json,
    compiled_plan_to_dict,
    compiled_plan_to_dto,
    compiled_plan_to_json,
)
from increment.errors import InvalidRequestError, WireFormatError
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.semantics.models import NormalPriorSpec
from increment.semantics.unit_cycle import UnitCycleTApproximation

_METRIC_A = "revenue_2024-01-15"
_METRIC_B = "orders_2024-01-16"


_GUARANTEE_BY_CORRECTION: dict[str, Literal["none", "fwer", "fdr"]] = {
    "none": "none",
    "bonferroni": "fwer",
    "bh": "fdr",
    "e_bh": "fdr",
}


def _policy(
    name: str,
    *,
    correction: Literal["none", "bh", "bonferroni", "e_bh"] = "none",
    q: float | None = None,
) -> MultiplicityFamily:
    return MultiplicityFamily(
        name=name,
        correction=correction,
        q=q,
        axes=("date", "metric"),
        guarantee=_GUARANTEE_BY_CORRECTION[correction],
    )


def _view_policies() -> CompiledViewPolicies:
    return CompiledViewPolicies(
        asof=_policy("asof_family", correction="bonferroni"),
        randomized_breakout=_policy("randomized_family", correction="bh", q=0.1),
        encouragement_breakout=_policy("encouragement_family"),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("bound_prior", "named_family", "expected_discovery"),
    [(True, True, True), (True, False, None), (False, True, None)],
    ids=["prior-exclusion", "explicit-no-family", "prior-free-exclusion"],
)
def test_portable_prior_reset_restores_declared_family(
    tmp_path, bound_prior, named_family, expected_discovery
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from increment import Analysis, AnalysisPlan, MetricSpec
    from tests.analysis_factory import lift_rows

    frame = pa.table(
        {
            "unit": range(40),
            "group": ["control"] * 20 + ["treatment"] * 20,
            "y": [10 + i % 3 for i in range(20)] + [20 + i % 3 for i in range(20)],
        }
    )
    prior = Normal(mu=0, sigma=0.1) if bound_prior else None
    source = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="group",
        control="control",
        metrics=[MetricSpec(name="y", prior=prior)],
        plan=AnalysisPlan(secondaries=["y"]),
    )
    oracle = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="group",
        control="control",
        metrics={"y": "mean"},
        plan=AnalysisPlan(secondaries=["y"]),
    )
    replay = None
    try:
        path = tmp_path / "moments.parquet"
        source.export(path)
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            payload = json.loads(row["decision_plan"])
            family = payload["procedures"]["y"]["family"]
            # Earlier exports persisted the prior's effective exclusion.
            family["member"] = False
            if not named_family:
                family["family"] = {"kind": "none"}
            row["decision_plan"] = json.dumps(payload)
        replay = Analysis.from_moments(
            rows, control="control", metrics=[MetricSpec(name="y", prior=prior)]
        )
        inherited = lift_rows(replay.run())[0]
        expected = lift_rows(oracle.run())[0].require_lift()
        for _ in range(2):
            cleared = lift_rows(replay.run(prior=None))[0]
            interval = cleared.require_lift()
            assert (interval.value, interval.lb, interval.ub) == pytest.approx(
                (expected.value, expected.lb, expected.ub)
            )
            assert cleared.discovery is expected_discovery
            assert cleared.family_axes == (
                ("metric", "arm") if expected_discovery is not None else None
            )
        restored = lift_rows(replay.run())[0]
        assert restored.discovery is None
        assert restored.require_lift().value == pytest.approx(inherited.require_lift().value)
        if bound_prior:
            assert restored.require_lift().value < expected.value
    finally:
        source.close()
        oracle.close()
        if replay is not None:
            replay.close()


def _rich_plan() -> CompiledDecisionPlan:
    inference = FixedInference()
    family = FamilyMembership(
        family=MultiplicityFamily(
            name="revenue_family",
            correction="bonferroni",
            axes=("date", "metric"),
            guarantee="fwer",
        ),
        member=True,
    )
    procedure = RelativeArmDecisionProcedure(
        metric=_METRIC_A,
        role="primary",
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        sensitivity_methods=(Method(name="unadjusted"),),
        methods_explicitly_empty=False,
        alternative="greater",
        prior=Normal(mu=0.02, sigma=0.15),
        prior_is_global=False,
        alpha=0.025,
        family=family,
        inference=inference,
        null_lift=0.0,
    )
    return CompiledDecisionPlan(
        declared=True,
        alpha=0.025,
        q=0.1,
        path="warehouse",
        inference=inference,
        procedures={_METRIC_A: procedure},
        view_policies=_view_policies(),
    )


def _arm_plan() -> CompiledDecisionPlan:
    return CompiledDecisionPlan(
        declared=True,
        alpha=0.05,
        q=0.1,
        path="warehouse",
        inference=FixedInference(),
        procedures={
            _METRIC_A: RelativeArmDecisionProcedure(
                metric=_METRIC_A,
                role="primary",
                decision_method=Method(name="unadjusted"),
                sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
                alternative="two-sided",
                prior=Normal(mu=0.0, sigma=0.2),
                alpha=0.05,
                null_lift=0.0,
                family=FamilyMembership(family=NoFamily()),
                inference=FixedInference(),
            ),
            _METRIC_B: AbsoluteArmDecisionProcedure(
                metric=_METRIC_B,
                role="secondary",
                decision_method=Method(name="unadjusted"),
                sensitivity_methods=(),
                methods_explicitly_empty=True,
                alternative="less",
                prior=None,
                alpha=0.05,
                null_abs=0.0,
                family=FamilyMembership(family=NoFamily()),
                inference=FixedInference(),
            ),
        },
        view_policies=_view_policies(),
    )


def _contrast_plan() -> CompiledDecisionPlan:
    return CompiledDecisionPlan(
        declared=True,
        alpha=0.05,
        q=0.1,
        path="frame/contrast",
        inference=FixedInference(),
        procedures={
            _METRIC_A: ContrastDecisionProcedure(
                reference=UnitCycleTApproximation(),
                metric=_METRIC_A,
                role="primary",
                alternative="greater",
                null_abs=0.0,
                alpha=0.05,
                preferred_direction="increase",
            )
        },
        view_policies=_view_policies(),
    )


@pytest.fixture(scope="module")
def dto_samples() -> tuple[BaseModel, ...]:
    family = WireMultiplicityFamily(
        name="family", correction="bonferroni", axes=("date", "metric"), guarantee="fwer"
    )
    membership = WireFamilyMembership(family=family, member=True)
    method = WireMethod(name="unadjusted", variance_reduction="none", conversion_inference="auto")
    from tests.sequential_cases import registration

    common: dict[str, Any] = {
        "metric": _METRIC_A,
        "role": "primary",
        "decision_method": method,
        "sensitivity_methods": (
            WireMethod(name="cuped", variance_reduction="cuped", conversion_inference="auto"),
        ),
        "alternative": "greater",
        "prior": NormalPriorSpec(mu=0.02, sigma=0.15),
        "prior_is_global": False,
        "alpha": 0.025,
        "family": membership,
        "inference": WireFixedInference(),
    }
    policies = WireViewPolicies(
        asof=family, randomized_breakout=family, encouragement_breakout=family
    )
    return (
        WireMethod(name="unadjusted", conversion_inference="auto"),
        WireFixedInference(),
        WireAlwaysValid(registration=registration()),
        WireNoFamily(),
        family,
        membership,
        WireRelativeArmProcedure(**common, null_lift=0.0),
        WireAbsoluteArmProcedure(
            **cast(Any, {**common, "inference": WireFixedInference(), "prior": None}),
            null_abs=0.0,
        ),
        WireContrastProcedure(
            metric=_METRIC_A,
            role="primary",
            alternative="greater",
            null_abs=0.0,
            alpha=0.025,
            preferred_direction="increase",
        ),
        policies,
        compiled_plan_to_dto(_rich_plan()),
        NormalPriorSpec(mu=0.02, sigma=0.15),
    )


def test_every_public_dto_round_trips_through_json_serialization(
    dto_samples: tuple[BaseModel, ...],
) -> None:
    for dto in dto_samples:
        dto_type: type[BaseModel] = type(dto)
        restored = dto_type.model_validate(dto.model_dump(mode="json"))
        assert restored == dto


def test_compiled_runtime_plans_round_trip_all_procedure_variants() -> None:
    for plan in (_arm_plan(), _contrast_plan(), _rich_plan()):
        assert compiled_plan_from_json(compiled_plan_to_json(plan)) == plan
        assert compiled_plan_from_dto(compiled_plan_to_dto(plan)) == plan


def test_prospective_contrast_reference_survives_compiled_wire():
    from tests.estimation.test_unit_cycle_envelope import envelope

    plan = _contrast_plan()
    reference = envelope(metric=_METRIC_A)
    procedure = cast(ContrastDecisionProcedure, plan.procedures[_METRIC_A]).model_copy(
        update={"reference": reference}
    )
    declared = plan.model_copy(update={"procedures": {_METRIC_A: procedure}})
    restored = compiled_plan_from_json(compiled_plan_to_json(declared))
    assert restored == declared


def test_golden_payload_has_stable_bytes() -> None:
    expected = (
        '{"alpha":0.025,"compliance":null,"declared":true,"inference":{"kind":"fixed"},"path":"warehouse",'
        '"procedures":{"revenue_2024-01-15":{"alpha":0.025,"alternative":"greater",'
        '"axis":"relative","decision_method":{"conversion_inference":"auto","name":"cuped",'
        '"variance_reduction":"cuped"},'
        '"family":{"family":{"axes":["date","metric"],"correction":"bonferroni",'
        '"guarantee":"fwer","kind":"multiplicity","name":"revenue_family","q":null},'
        '"member":true},"inference":{"kind":"fixed"},"kind":"relative",'
        '"methods_explicitly_empty":false,"metric":"revenue_2024-01-15","null_lift":0.0,'
        '"prior":{"mu":0.02,"sigma":0.15},"prior_is_global":false,"role":"primary",'
        '"scale":"relative","sensitivity_methods":[{"conversion_inference":"auto",'
        '"name":"unadjusted","variance_reduction":"none"}]}},'
        '"q":0.1,"view_policies":{"asof":{"axes":["date","metric"],"correction":"bonferroni",'
        '"guarantee":"fwer","kind":"multiplicity","name":"asof_family","q":null},'
        '"encouragement_breakout":{"axes":["date","metric"],"correction":"none",'
        '"guarantee":"none","kind":"multiplicity","name":"encouragement_family","q":null},'
        '"randomized_breakout":{"axes":["date","metric"],"correction":"bh",'
        '"guarantee":"fdr","kind":"multiplicity","name":"randomized_family","q":0.1}},"wire_version":2}'
    )
    payload = compiled_plan_to_json(_rich_plan())
    assert payload == expected
    assert compiled_plan_from_json(expected) == _rich_plan()


def _finite_sample_plan() -> CompiledDecisionPlan:
    plan = _arm_plan()
    procedure = cast(RelativeArmDecisionProcedure, plan.procedures[_METRIC_A]).model_copy(
        update={
            "decision_method": Method(name="unadjusted", conversion_inference="finite_sample"),
            "prior": None,
            "sensitivity_methods": (),
        }
    )
    return plan.model_copy(update={"procedures": {**plan.procedures, _METRIC_A: procedure}})


def test_conversion_inference_is_always_emitted_and_survives_the_wire():
    plan = _finite_sample_plan()
    payload = cast(dict[str, Any], compiled_plan_to_dict(plan))
    assert payload["procedures"][_METRIC_A]["decision_method"]["conversion_inference"] == (
        "finite_sample"
    )
    assert payload["procedures"][_METRIC_B]["decision_method"]["conversion_inference"] == "auto"
    for restored in (
        compiled_plan_from_dict(payload),
        compiled_plan_from_json(compiled_plan_to_json(plan)),
        compiled_plan_from_dto(compiled_plan_to_dto(plan)),
    ):
        assert restored == plan
        method = cast(RelativeArmDecisionProcedure, restored.procedures[_METRIC_A]).decision_method
        assert method.conversion_inference == "finite_sample"


def _legacy_payload(plan: CompiledDecisionPlan) -> dict[str, Any]:
    """The payload an earlier version wrote: no ``conversion_inference`` on any method."""
    payload = cast(dict[str, Any], compiled_plan_to_dict(plan))
    for procedure in payload["procedures"].values():
        for method in (procedure["decision_method"], *procedure["sensitivity_methods"]):
            del method["conversion_inference"]
    return payload


def _legacy_unadjusted_plan(
    metric_type: str | None = None, method: Method | None = None
) -> dict[str, Any]:
    """A legacy payload of one prior-free, fixed-horizon, relative-scale procedure whose decision
    method is ``method`` (the unadjusted ``Method(name="unadjusted")`` by default)."""
    plan = _arm_plan()
    procedure = cast(RelativeArmDecisionProcedure, plan.procedures[_METRIC_A]).model_copy(
        update={
            "prior": None,
            "sensitivity_methods": (),
            **({} if method is None else {"decision_method": method}),
        }
    )
    return _legacy_payload(plan.model_copy(update={"procedures": {_METRIC_A: procedure}}))


def _decision_method(restored: CompiledDecisionPlan) -> Method:
    return cast(RelativeArmDecisionProcedure, restored.procedures[_METRIC_A]).decision_method


def test_a_legacy_unadjusted_method_decodes_to_the_route_it_ran():
    """Before the field existed every unadjusted conversion row of a fixed-horizon, prior-free
    procedure ran the finite-sample route: a stored plan decodes to that, never silently to
    ``auto``."""
    restored = compiled_plan_from_dict(_legacy_unadjusted_plan())
    assert _decision_method(restored) == Method(
        name="unadjusted", conversion_inference="finite_sample"
    )


@pytest.mark.parametrize("label", ["custom", "control_v2", "ols"])
def test_a_legacy_free_form_method_label_replays_on_the_route_it_ran(label):
    """``Method.name`` is a free-form label: the historical randomized estimator sent every
    non-CUPED conversion row of a fixed-horizon, prior-free procedure through the binomial set,
    so a stored method under any other label decodes and replays there, not on ``auto``'s
    delta method."""
    restored = compiled_plan_from_dict(
        _legacy_unadjusted_plan(method=Method(name=label)), metric_types={_METRIC_A: "conversion"}
    )
    assert _decision_method(restored) == Method(name=label, conversion_inference="finite_sample")
    assert _executed_row_kinds(restored, "conversion") == ["binomial"]
    assert _method_row_kinds(Method(name=label), "conversion") == ["t"]


@pytest.mark.parametrize(
    "method",
    [
        Method(name="cuped", variance_reduction="cuped"),
        Method(name="adjusted", variance_reduction="cuped"),
        Method(name="iptw"),
        Method(name="aipw"),
        Method(name="dml"),
    ],
    ids=lambda method: f"{method.name}-{method.variance_reduction}",
)
def test_a_legacy_adjusted_method_stays_on_auto_whatever_its_label(method):
    """A CUPED method (under any label) and an observational estimator never took the binomial
    route, so a stored one must not acquire ``finite_sample``."""
    restored = compiled_plan_from_dict(
        _legacy_unadjusted_plan(method=method), metric_types={_METRIC_A: "conversion"}
    )
    assert _decision_method(restored) == method
    assert _decision_method(restored).conversion_inference == "auto"


@pytest.mark.parametrize("metric_type", ["conversion", "retention"])
def test_a_legacy_conversion_or_retention_metric_decodes_to_finite_sample(metric_type):
    restored = compiled_plan_from_dict(
        _legacy_unadjusted_plan(), metric_types={_METRIC_A: metric_type}
    )
    assert _decision_method(restored).conversion_inference == "finite_sample"


@pytest.mark.parametrize("metric_type", ["mean", "ratio", "quantile", "total"])
def test_a_legacy_method_on_a_metric_without_that_route_decodes_to_auto(metric_type):
    """A mean, ratio, quantile or total metric never had the finite-sample route, and an
    explicit ``finite_sample`` on it is refused at runtime: the stored plan must not acquire it."""
    restored = compiled_plan_from_dict(
        _legacy_unadjusted_plan(), metric_types={_METRIC_A: metric_type}
    )
    assert _decision_method(restored).conversion_inference == "auto"


def test_a_legacy_plan_with_a_prior_or_an_adjustment_decodes_to_auto():
    """Each of these never took the binomial route, and explicit ``finite_sample`` refuses them."""
    restored = compiled_plan_from_dict(_legacy_payload(_arm_plan()))
    first = cast(RelativeArmDecisionProcedure, restored.procedures[_METRIC_A])
    assert first.prior is not None
    assert first.decision_method.conversion_inference == "auto"
    # A CUPED sensitivity method is variance reduction, never the binomial route.
    assert first.sensitivity_methods == (Method(name="cuped", variance_reduction="cuped"),)


def test_a_legacy_absolute_margin_plan_decodes_to_the_route_it_ran():
    """A prior-free absolute-margin procedure read the same binomial set as a relative one: its
    stored method is ``finite_sample``, not ``auto``."""
    restored = compiled_plan_from_dict(_legacy_payload(_arm_plan()))
    second = cast(AbsoluteArmDecisionProcedure, restored.procedures[_METRIC_B])
    assert second.decision_method == Method(name="unadjusted", conversion_inference="finite_sample")


def test_a_wire_method_must_state_its_route():
    with pytest.raises(InvalidRequestError) as raised:
        WireMethod.model_validate({"name": "unadjusted"})
    assert raised.value.code == "model.field.missing"


def _method_row_kinds(
    method: Method, metric_type: str, *, null_abs: float | None = None
) -> list[str | None]:
    """The ``reference_kind`` of ``method`` run through ``estimate_lift`` on a dense contrast of
    a metric of ``metric_type``, against an absolute margin when ``null_abs`` is given."""
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import ConversionMetric, MeanMetric
    from tests.estimation._conversion_counts import count_summary

    metric = (
        ConversionMetric(name="conv", entity="user", fact="conv")
        if metric_type == "conversion"
        else MeanMetric(name="conv", entity="user", fact="conv")
    )
    computation = estimate_lift(
        metrics=[metric],
        summary=count_summary(50_000, 1_000_000, 51_500, 1_000_000),
        control_group="control",
        methods=[method],
        null_abs=null_abs,
    )
    return [row.reference_kind for row in computation.results]


def _executed_row_kinds(restored: CompiledDecisionPlan, metric_type: str) -> list[str | None]:
    """The ``reference_kind`` of the decoded decision method run on a dense contrast."""
    return _method_row_kinds(_decision_method(restored), metric_type)


def test_a_legacy_plan_replays_on_every_metric_type_it_could_be_stored_for():
    """The decoded plan runs: a conversion metric on the finite-sample route it always took (not
    on ``auto``'s delta method), and a mean metric on its ordinary route without the refusal an
    explicit ``finite_sample`` would raise."""
    types = {_METRIC_A: "conversion"}
    conversion = compiled_plan_from_dict(_legacy_unadjusted_plan(), metric_types=types)
    assert _executed_row_kinds(conversion, "conversion") == ["binomial"]
    mean = compiled_plan_from_dict(_legacy_unadjusted_plan(), metric_types={_METRIC_A: "mean"})
    assert _executed_row_kinds(mean, "mean") == ["t"]


def test_a_legacy_absolute_margin_plan_replays_on_the_route_it_ran():
    """A dense absolute-margin conversion decision stored before the field existed ran the
    finite-sample route, so it replays there and not on ``auto``'s delta method."""
    restored = compiled_plan_from_dict(_legacy_payload(_arm_plan()))
    procedure = cast(AbsoluteArmDecisionProcedure, restored.procedures[_METRIC_B])
    method, margin = procedure.decision_method, procedure.null_abs
    assert _method_row_kinds(method, "conversion", null_abs=margin) == ["binomial"]
    assert _method_row_kinds(Method(name="unadjusted"), "conversion", null_abs=margin) == ["t"]


def _lift_kinds(rows) -> dict[str, str | None]:
    from increment.estimation.results import LiftEstimate

    return {row.metric: row.reference_kind for row in rows if isinstance(row, LiftEstimate)}


def test_a_cube_exported_before_the_field_existed_replays_on_every_metric_type(tmp_path):
    """The stored-plan seam end to end: an exported moments cube whose embedded plan lacks
    ``conversion_inference`` rehydrates and runs. Its conversion metric keeps the finite-sample
    route its plan was written under (a new default plan would take the delta method on these
    dense counts), and its mean metric runs on its ordinary route instead of being refused as an
    explicit ``finite_sample`` on a metric that has none."""
    import json

    import pandas as pd
    import pyarrow.parquet as pq

    from increment import Analysis
    from increment.frame import MetricSpec

    n = 10_000
    frame = pd.DataFrame(
        {
            "unit_id": [f"{group}{i}" for group in ("control", "treatment") for i in range(n)],
            "group_id": ["control"] * n + ["treatment"] * n,
            "conv": [int(i < 3_000) for i in range(n)] + [int(i < 3_150) for i in range(n)],
            "rev": [float(1 + i % 7) for i in range(n)]
            + [float(1 + (i + 1) % 7) for i in range(n)],
        }
    )
    metrics = [MetricSpec(name="conv", type="conversion"), MetricSpec(name="rev", type="mean")]
    current = Analysis.from_unit_summary(
        frame, unit="unit_id", group="group_id", control="control", metrics=metrics
    )
    path = tmp_path / "cube.parquet"
    current.export(path)
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        stored = json.loads(row["decision_plan"])
        for procedure in stored["procedures"].values():
            for method in (procedure["decision_method"], *procedure["sensitivity_methods"]):
                del method["conversion_inference"]
        row["decision_plan"] = json.dumps(stored)
    replay = Analysis.from_moments(rows, metrics=metrics, control="control")
    assert _lift_kinds(replay.run()) == {"conv": "binomial", "rev": "t"}
    assert _lift_kinds(current.run()) == {"conv": "t", "rev": "t"}


def test_the_same_counts_under_a_new_default_method_take_the_delta_method_route():
    from increment.estimation.engine import estimate_lift
    from tests.estimation._conversion_counts import CONVERSION_METRIC, count_summary

    current = estimate_lift(
        metrics=[CONVERSION_METRIC],
        summary=count_summary(50_000, 1_000_000, 51_500, 1_000_000),
        control_group="control",
        methods=[Method(name="unadjusted")],
    )
    assert [row.reference_kind for row in current.results] == ["t"]


def test_an_unknown_conversion_inference_is_rejected():
    payload = cast(dict[str, Any], compiled_plan_to_dict(_arm_plan()))
    payload["procedures"][_METRIC_A]["decision_method"]["conversion_inference"] = "asymptotic"
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_from_dict(payload)
    assert raised.value.code == "wire.payload.invalid"


def test_wire_dto_procedures_support_copy_and_pickle_without_mutability():
    dto = compiled_plan_to_dto(_rich_plan())
    for cloned in (copy.deepcopy(dto), pickle.loads(pickle.dumps(dto)), dto.model_copy(deep=True)):
        assert cloned.model_dump(mode="json") == dto.model_dump(mode="json")
        with pytest.raises(TypeError):
            cast(Any, cloned.procedures)["new"] = dto.procedures[next(iter(dto.procedures))]


@pytest.mark.parametrize(
    "update",
    [{"alpha": 0}, {"alpha": "invalid"}, {"path": 3}, {"procedures": None}],
)
def test_compiled_plan_from_dto_wraps_invalid_bypass_fields(update: dict[str, object]):
    good = compiled_plan_to_dto(_rich_plan())
    bypassed = WireCompiledDecisionPlan.model_construct(
        **cast(Any, {**good.model_dump(), **update})
    )
    with pytest.raises(WireFormatError) as exc:
        compiled_plan_from_dto(bypassed)
    assert exc.value.code == "wire.payload.invalid"


def _golden_asymptotic_plan() -> CompiledDecisionPlan:
    from fractions import Fraction

    from increment.frame import MetricSpec, synthesise_metric
    from increment.plan import bind_automatic_sequential_plan, compile_decision_plan
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
    from increment.semantics.models import AnalysisPlan, InferenceSpec
    from increment.semantics.sequential import SequentialCompliancePolicy
    from increment.sequential_source import frame_observation_mapping

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True,
            justification="the prompt affects revenue only through clicks",
        ),
        allocation={"control": 0.5, "treatment": 0.5},
    )
    specs = [MetricSpec(name="revenue", type="mean"), MetricSpec(name="orders", type="mean")]
    metrics = [synthesise_metric(spec) for spec in specs]
    plan = AnalysisPlan(
        primary="revenue",
        inference=InferenceSpec(kind="asymptotic_mean"),
        compliance=SequentialCompliancePolicy(alpha=Fraction(1, 20)),
    )
    bound = bind_automatic_sequential_plan(
        plan,
        metrics,
        design=design,
        source_id="frame",
        source_mapping=frame_observation_mapping(unit="unit", group="arm", uptake="clicked"),
        transformations=specs,
        path="frame",
    )
    assert bound is not None
    assert bound.inference is not None
    assert bound.inference.registration is not None
    estimands = tuple(sorted({cell.estimand for cell in bound.inference.registration.roster}))
    return compile_decision_plan(
        bound,
        metrics,
        path="frame",
        design=design,
        estimands=estimands,
    )


def test_decision_wire_goldens_round_trip_byte_for_byte():

    variants = {
        "fixed_relative_multiplicity_family": _rich_plan(),
        "fixed_relative_and_absolute_mixed_no_family": _arm_plan(),
        "fixed_frame_contrast": _contrast_plan(),
        "asymptotic_mean_mixed_family": _golden_asymptotic_plan(),
    }
    root = Path(__file__).parent / "goldens" / "decision_wire"
    for name, plan in variants.items():
        expected = (root / f"{name}.json").read_text()
        encoded = compiled_plan_to_json(plan)
        assert json.dumps(json.loads(encoded), indent=2, sort_keys=True) + "\n" == expected
        restored = compiled_plan_from_json(expected)
        assert compiled_plan_to_dict(restored) == compiled_plan_to_dict(plan)


def test_procedures_mapping_on_a_validated_dto_is_immutable():
    """A frozen WireCompiledDecisionPlan must not let callers mutate procedures."""
    dto = compiled_plan_to_dto(_rich_plan())
    metric = next(iter(dto.procedures))
    with pytest.raises(TypeError):
        cast(Any, dto.procedures)[metric] = dto.procedures[metric]


def test_compiled_plan_from_dto_revalidates_a_bypass_constructed_dto():
    """DTO instances built with model_construct still undergo wire validation."""
    good = compiled_plan_to_dto(_rich_plan())
    metric = next(iter(good.procedures))
    bad_procedure = good.procedures[metric].model_copy(update={"alpha": 0.5})
    bypassed = WireCompiledDecisionPlan.model_construct(
        **cast(Any, {**good.model_dump(), "procedures": {metric: bad_procedure}})
    )
    with pytest.raises(WireFormatError) as exc:
        compiled_plan_from_dto(bypassed)
    assert exc.value.code == "wire.compiled_plan.procedure_alpha_mismatch"


def test_numeric_nulls_remain_null_and_missing_defaults_are_emitted() -> None:
    payload = cast(dict[str, Any], compiled_plan_to_dict(_rich_plan()))
    assert payload["procedures"][_METRIC_A]["family"]["family"]["q"] is None
    restored = compiled_plan_from_dict(payload)
    assert cast(Any, restored.procedures[_METRIC_A].family).family.q is None

    omitted = json.loads(compiled_plan_to_json(_rich_plan()))
    del omitted["procedures"][_METRIC_A]["family"]["family"]["q"]
    restored_omitted = compiled_plan_from_dict(omitted)
    assert cast(Any, restored_omitted.procedures[_METRIC_A].family).family.q is None


def test_collection_order_follows_mapping_sorting_and_tuple_order() -> None:
    payload = cast(dict[str, Any], compiled_plan_to_dict(_arm_plan()))
    assert list(payload["procedures"]) == [_METRIC_A, _METRIC_B]
    assert [
        method["name"] for method in payload["procedures"][_METRIC_A]["sensitivity_methods"]
    ] == ["cuped"]
    assert payload["procedures"][_METRIC_A]["family"]["family"] == {"kind": "none"}

    raw = json.loads(compiled_plan_to_json(_arm_plan()))
    assert list(raw["procedures"]) == sorted([_METRIC_A, _METRIC_B])
    assert list(raw["view_policies"]) == ["asof", "encouragement_breakout", "randomized_breakout"]
    rich_payload = cast(dict[str, Any], compiled_plan_to_dict(_rich_plan()))
    assert rich_payload["procedures"][_METRIC_A]["family"]["family"]["axes"] == ["date", "metric"]


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda payload: payload.update(unexpected=True), "wire.payload.invalid"),
        (lambda payload: payload.update(path=3), "wire.payload.invalid"),
        (lambda payload: payload.update(procedures=[]), "wire.payload.invalid"),
    ],
)
def test_invalid_dict_payloads_are_rejected(
    mutate: Callable[[dict[str, object]], None], code: str
) -> None:
    payload = compiled_plan_to_dict(_rich_plan())
    mutate(payload)
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_from_dict(payload)
    assert raised.value.code == code


def test_duplicate_json_keys_are_rejected_at_any_object_level() -> None:
    payload = compiled_plan_to_json(_rich_plan())
    duplicate_top_level = payload.replace('"declared":true', '"declared":true,"declared":true', 1)
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_from_json(duplicate_top_level)
    assert raised.value.code == "wire.payload.duplicate_key"

    duplicate_nested = payload.replace('"name":"cuped"', '"name":"cuped","name":"cuped"', 1)
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_from_json(duplicate_nested)
    assert raised.value.code == "wire.payload.duplicate_key"


def test_non_object_json_payload_is_rejected() -> None:
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_from_json("[]")
    assert raised.value.code == "wire.payload.invalid"


def _change_procedure_inference(payload: dict[str, Any]) -> None:
    from tests.sequential_cases import registration

    payload["procedures"][_METRIC_A]["inference"] = {
        "kind": "always_valid",
        "registration": registration().model_dump(mode="json"),
    }


@pytest.mark.parametrize(
    ("plan_fn", "mutate", "code"),
    [
        (
            _rich_plan,
            lambda payload: payload["procedures"][_METRIC_A].update(alpha=0.05),
            "wire.compiled_plan.procedure_alpha_mismatch",
        ),
        (
            _rich_plan,
            _change_procedure_inference,
            "wire.compiled_plan.procedure_inference_mismatch",
        ),
        (
            _rich_plan,
            lambda payload: payload.update(path="frame/contrast"),
            "wire.compiled_plan.frame_contrast_requires_contrast",
        ),
        (
            _contrast_plan,
            lambda payload: payload.update(path="warehouse"),
            "wire.compiled_plan.arm_plan_requires_arm",
        ),
    ],
)
def test_compiled_invariant_violations_are_rejected_with_stable_code(
    plan_fn: Callable[[], CompiledDecisionPlan],
    mutate: Callable[[dict[str, object]], None],
    code: str,
) -> None:
    payload = cast(dict[str, Any], compiled_plan_to_dict(plan_fn()))
    mutate(payload)
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_from_dict(payload)
    assert raised.value.code == code


def test_wire_compiled_decision_plan_direct_construction_carries_code() -> None:
    payload = cast(dict[str, Any], compiled_plan_to_dict(_rich_plan()))
    payload["procedures"][_METRIC_A]["alpha"] = 0.05
    with pytest.raises(WireFormatError) as raised:
        WireCompiledDecisionPlan(**payload)
    assert raised.value.code == "wire.compiled_plan.procedure_alpha_mismatch"


def test_compiled_plan_ambiguous_union_failure_stays_validation_error() -> None:
    plan = _contrast_plan()
    runtime = dict(plan)
    runtime["procedures"] = {key: value.model_dump() for key, value in plan.procedures.items()}
    first = next(iter(runtime["procedures"].values()))
    first["alpha"] = 0
    with pytest.raises(ValidationError):
        CompiledDecisionPlan.model_validate(runtime)


@pytest.mark.parametrize(
    "key",
    ["relative", "absolute", "WireRelativeArmProcedure", "WireAbsoluteArmProcedure", "m[a]"],
)
def test_wire_procedure_key_never_stands_in_for_a_union_member(key: str) -> None:
    """Mapping keys are consumed as keys even when they read like tags or labels."""
    payload = cast(dict[str, Any], compiled_plan_to_dict(_arm_plan()))
    relative = payload["procedures"].pop(_METRIC_A)
    payload["procedures"] = {key: relative, **payload["procedures"]}

    relative["alpha"] = 0
    with pytest.raises(InvalidRequestError) as raised:
        WireCompiledDecisionPlan.model_validate(payload)
    assert raised.value.code == "model.field.range"
    assert raised.value.context["model"] == "WireRelativeArmProcedure"
    assert raised.value.context["field"] == "alpha"

    relative["alpha"] = 0.05
    relative["family"]["family"]["correction"] = "bonferroni"
    with pytest.raises(InvalidRequestError) as raised:
        WireCompiledDecisionPlan.model_validate(payload)
    assert raised.value.code == "model.field.unknown"
    assert raised.value.context["model"] == "WireNoFamily"
    assert raised.value.context["field"] == "correction"


def test_wire_procedure_keys_named_after_members_do_not_read_as_ambiguous() -> None:
    payload = cast(dict[str, Any], compiled_plan_to_dict(_arm_plan()))
    relative = payload["procedures"][_METRIC_A]
    relative["alpha"] = 0
    payload["procedures"] = {
        "WireRelativeArmProcedure": relative,
        "WireAbsoluteArmProcedure": copy.deepcopy(relative),
    }
    with pytest.raises(InvalidRequestError) as raised:
        WireCompiledDecisionPlan.model_validate(payload)
    assert raised.value.code == "model.field.range"
    assert raised.value.context["model"] == "WireRelativeArmProcedure"
    assert raised.value.context["field"] == "alpha"


def test_wire_procedure_unknown_key_that_matches_a_literal_is_still_unknown() -> None:
    payload = cast(dict[str, Any], compiled_plan_to_dict(_arm_plan()))
    payload["procedures"][_METRIC_A]["primary"] = True
    with pytest.raises(InvalidRequestError) as raised:
        WireCompiledDecisionPlan.model_validate(payload)
    assert raised.value.code == "model.field.unknown"
    assert raised.value.context["model"] == "WireRelativeArmProcedure"
    assert raised.value.context["field"] == "primary"


def test_wire_inference_unknown_key_is_attributed_to_the_tagged_member() -> None:
    payload = cast(dict[str, Any], compiled_plan_to_dict(_arm_plan()))
    payload["inference"]["extra"] = 1
    with pytest.raises(InvalidRequestError) as raised:
        WireCompiledDecisionPlan.model_validate(payload)
    assert raised.value.code == "model.field.unknown"
    assert raised.value.context["model"] == "WireFixedInference"
    assert raised.value.context["field"] == "extra"


@pytest.mark.parametrize(
    ("correction", "guarantee"),
    [("none", "fwer"), ("bonferroni", "fdr"), ("bh", "none"), ("e_bh", "none")],
)
def test_wire_multiplicity_family_rejects_guarantee_mismatch(correction, guarantee) -> None:
    q = 0.1 if correction in ("bh", "e_bh") else None
    with pytest.raises(WireFormatError) as exc:
        WireMultiplicityFamily(name="f", correction=correction, q=q, guarantee=guarantee)
    assert exc.value.code == "wire.multiplicity.guarantee_mismatch"
    assert exc.value.context == {"correction": correction, "guarantee": guarantee}


@pytest.mark.parametrize("correction", ["bh", "e_bh"])
def test_wire_multiplicity_family_rejects_bh_without_q(correction) -> None:
    with pytest.raises(WireFormatError) as exc:
        WireMultiplicityFamily(name="f", correction=correction, q=None, guarantee="fdr")
    assert exc.value.code == "wire.multiplicity.validate_policy"


def test_wire_multiplicity_family_rejects_q_outside_bh() -> None:
    with pytest.raises(WireFormatError) as exc:
        WireMultiplicityFamily(name="f", correction="none", q=0.1, guarantee="none")
    assert exc.value.code == "wire.multiplicity.bh"


def test_compiled_plan_from_dict_rejects_mutated_guarantee() -> None:
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    payload = cast(dict[str, Any], compiled_plan_to_dict(compile_decision_plan(None, [metric])))
    policy = cast(dict[str, Any], payload["view_policies"])["randomized_breakout"]
    policy["correction"] = "none"
    policy["q"] = None
    policy["guarantee"] = "fwer"

    with pytest.raises(WireFormatError) as exc:
        compiled_plan_from_dict(payload)
    assert exc.value.code == "wire.multiplicity.guarantee_mismatch"
