from __future__ import annotations

import copy
import pickle

import pytest
from pydantic import ValidationError

from increment.analysis import Analysis
from increment.errors import DefinitionError, InvalidRequestError
from increment.estimation.cuped import fit_cuped_multicovariate
from increment.estimation.engine import Method
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec
from increment.semantics.models import (
    CupedCovariate,
    CupedCovariateBinding,
    ExperimentMetric,
    Measure,
    MethodSpec,
)


def test_semantic_cuped_covariate_types_and_method_are_constructible():
    covariate = CupedCovariate(name="country", kind="categorical")
    binding = CupedCovariateBinding(
        name="pre_revenue", kind="continuous", measure=Measure(fact="purchase", aggregation="sum")
    )

    assert covariate.missing == "refuse"
    assert binding.measure is not None
    assert (
        MethodSpec(
            name="linear",
            variance_reduction="cuped_multicovariate",
            covariates=(covariate,),
        ).variance_reduction
        == "cuped_multicovariate"
    )


def test_method_factories_normalize_covariates_and_roundtrip_binding():
    scalar_covariate = CupedCovariate(name="prior", kind="continuous", missing="impute")
    mixed_covariate = CupedCovariate(name="prior", kind="continuous", missing="refuse")
    binding = CupedCovariateBinding(
        name="pre_revenue",
        kind="continuous",
        missing="refuse",
        measure=Measure(fact="purchase", aggregation="sum"),
    )
    mixed = Method.cuped_multicovariate(covariates=[mixed_covariate, binding])
    scalar = Method.cuped(covariates=scalar_covariate)
    assert scalar.variance_reduction == "cuped"
    assert scalar.covariates == (scalar_covariate,)
    assert Method.cuped().covariates is None
    assert mixed.variance_reduction == "cuped_multicovariate"
    assert mixed.covariates is not None
    assert mixed.covariates[1].model_dump()["measure"] == binding.model_dump()["measure"]
    declaration = MethodSpec(
        name=mixed.name,
        variance_reduction=mixed.variance_reduction,
        covariates=mixed.covariates,
    )
    assert declaration.covariates == mixed.covariates
    assert declaration.model_dump()["covariates"][1]["measure"] == binding.model_dump()["measure"]


def test_method_factories_enforce_procedure_specific_covariates():
    with pytest.raises(InvalidRequestError):
        Method.cuped_multicovariate(covariates=[])
    with pytest.raises(InvalidRequestError):
        Method.cuped(
            covariates=[
                CupedCovariate(name="prior", kind="continuous"),
                CupedCovariate(name="region", kind="categorical"),
            ]
        )
    with pytest.raises(InvalidRequestError):
        Method.cuped(covariates=CupedCovariate(name="region", kind="categorical"))


def test_metric_spec_no_longer_accepts_metric_owned_cuped_covariates():
    with pytest.raises(InvalidRequestError) as raised:
        MetricSpec(
            name="revenue",
            covariate="prior",  # ty: ignore[unknown-argument]
            covariate_missing="zero",  # ty: ignore[unknown-argument]
            covariates=(CupedCovariate(name="prior", kind="continuous"),),  # ty: ignore[unknown-argument]
        )
    assert raised.value.code == "model.field.unknown"


def test_method_covariates_preserve_the_implicit_scalar_marker():
    from types import SimpleNamespace

    from increment._analysis_config import method_covariates

    config = SimpleNamespace()
    assert method_covariates(config, Method.cuped()) is None
    covariate = CupedCovariate(name="prior", kind="continuous")
    assert method_covariates(config, Method.cuped(covariates=covariate)) == (covariate,)
    assert method_covariates(config, Method.unadjusted()) == ()


def test_frame_plan_method_overrides_and_explicit_empty_sensitivity() -> None:
    import pyarrow as pa

    from increment.semantics.models import AnalysisPlan

    frame = pa.table(
        {
            "unit": ["c1", "c2", "t1", "t2"],
            "arm": ["control", "control", "treatment", "treatment"],
            "revenue": [10.0, 12.0, 13.0, 15.0],
            "prior": [2.0, 3.0, 4.0, 5.0],
        }
    )
    declaration = MethodSpec(
        name="planned",
        variance_reduction="cuped",
        covariates=(CupedCovariate(name="prior", kind="continuous"),),
    )
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="revenue",
            decision_method=declaration,
            sensitivity_methods=(),
        )
    )
    metric = MetricSpec(
        name="revenue",
        sensitivity_methods=(Method.unadjusted(name="inherited"),),
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[metric],
        plan=plan,
    )

    rows = analysis.run()
    assert [row.method for row in rows] == ["planned"]


def test_covariate_union_allows_policy_differences_but_rejects_binding_conflicts():
    from types import SimpleNamespace

    from increment._analysis_config import union_covariates

    scalar = Method.cuped(
        covariates=CupedCovariate(name="prior", kind="continuous", missing="impute"),
        name="scalar",
    )
    mixed = Method.cuped_multicovariate(
        covariates=CupedCovariate(name="prior", kind="continuous", missing="refuse"),
        name="mixed",
    )
    (resolved,) = union_covariates([SimpleNamespace()], [(scalar, mixed)])
    assert resolved.name == "prior"

    bound = CupedCovariateBinding(name="prior", kind="continuous", property="prior_a")
    other = CupedCovariateBinding(name="prior", kind="continuous", property="prior_b")
    with pytest.raises(InvalidRequestError):
        union_covariates(
            [SimpleNamespace()],
            [
                (
                    Method.cuped_multicovariate(covariates=bound, name="one"),
                    Method.cuped_multicovariate(covariates=other, name="two"),
                )
            ],
        )


@pytest.mark.parametrize(
    "aggregation",
    ["sum", "count", "count_distinct", "avg_event", "avg_calendar_day", "min", "max"],
)
def test_measure_covariates_accept_numeric_measure_aggregations(aggregation):
    binding = CupedCovariateBinding(
        name="pre_value",
        kind="continuous",
        measure=Measure.model_validate({"fact": "purchase", "aggregation": aggregation}),
    )

    assert binding.measure is not None
    assert binding.measure.aggregation == aggregation


def test_public_run_routes_mixed_covariate_fit():
    import numpy as np
    import pyarrow as pa

    frame = pa.table(
        {
            "unit": [f"{arm}{i}" for arm in ("c", "t") for i in range(16)],
            "arm": ["c"] * 16 + ["t"] * 16,
            "revenue": [float(i + (arm == "t") * 2) for arm in ("c", "t") for i in range(16)],
            "prior": [float(i % 4) for _arm in ("c", "t") for i in range(16)],
            "region": ["north", "north", "south", "south"] * 8,
        }
    )
    metric = MetricSpec(
        name="revenue",
        decision_method=Method.cuped_multicovariate(
            name="linear",
            covariates=(
                CupedCovariate(name="prior", kind="continuous"),
                CupedCovariate(name="region", kind="categorical"),
            ),
        ),
    )
    analysis = Analysis.from_unit_summary(
        frame, unit="unit", group="arm", control="c", metrics=[metric]
    )

    (row,) = analysis.run()
    assert isinstance(row, LiftEstimate)
    expected = (
        fit_cuped_multicovariate(
            np.asarray(frame["revenue"].to_pylist()),
            frame["arm"].to_pylist(),
            control="c",
            arm_order=("c", "t"),
            covariates={
                "prior": np.asarray(frame["prior"].to_pylist()),
                "region": np.asarray(frame["region"].to_pylist(), dtype=object),
            },
            kinds={"prior": "continuous", "region": "categorical"},
            missing={"prior": "refuse", "region": "refuse"},
        )
        .relative("t")
        .estimate
    )

    assert row.require_lift().value == pytest.approx(np.expm1(expected))


def test_public_run_routes_joint_ratio_cuped_projection():
    import numpy as np
    import pyarrow as pa

    frame = pa.table(
        {
            "unit": [f"{arm}{i}" for arm in ("c", "t") for i in range(16)],
            "arm": ["c"] * 16 + ["t"] * 16,
            "revenue": [float(4 + i + (arm == "t") * 2) for arm in ("c", "t") for i in range(16)],
            "orders": [float(2 + i % 3 + (arm == "t")) for arm in ("c", "t") for i in range(16)],
            "prior": [float(i % 4) for _arm in ("c", "t") for i in range(16)],
            "region": ["north", "north", "south", "south"] * 8,
        }
    )
    metric = MetricSpec(
        name="aov",
        type="ratio",
        numerator="revenue",
        denominator="orders",
        decision_method=Method.cuped_multicovariate(
            name="linear",
            covariates=(
                CupedCovariate(name="prior", kind="continuous"),
                CupedCovariate(name="region", kind="categorical"),
            ),
        ),
    )
    analysis = Analysis.from_unit_summary(
        frame, unit="unit", group="arm", control="c", metrics=[metric]
    )

    result = analysis.run()
    (row,) = result
    assert isinstance(row, LiftEstimate)
    expected = (
        fit_cuped_multicovariate(
            None,
            frame["arm"].to_pylist(),
            control="c",
            arm_order=("c", "t"),
            covariates={
                "prior": np.asarray(frame["prior"].to_pylist()),
                "region": np.asarray(frame["region"].to_pylist(), dtype=object),
            },
            kinds={"prior": "continuous", "region": "categorical"},
            missing={"prior": "refuse", "region": "refuse"},
            numerator=np.asarray(frame["revenue"].to_pylist()),
            denominator=np.asarray(frame["orders"].to_pylist()),
        )
        .ratio("t", scale="log")
        .estimate
    )

    assert row.require_lift().value == pytest.approx(np.expm1(expected))

    reissued = row.reintervalize(0.01)
    assert reissued.require_lift().value == pytest.approx(row.require_lift().value)
    new_lift = reissued.require_lift()
    old_lift = row.require_lift()
    assert new_lift.ub is not None and new_lift.lb is not None
    assert old_lift.ub is not None and old_lift.lb is not None
    assert new_lift.ub - new_lift.lb > old_lift.ub - old_lift.lb
    assert result.metadata is not None
    assert any(
        advisory.code == "estimation.cuped_linear.fit"
        for record in result.metadata.cells
        for advisory in record.advisories
    )

    shifted = frame.set_column(
        frame.schema.get_field_index("prior"),
        "prior",
        pa.array([value + 1.0 for value in frame["prior"].to_pylist()]),
    ).set_column(
        frame.schema.get_field_index("region"),
        "region",
        pa.array(
            ["south" if value == "north" else "north" for value in frame["region"].to_pylist()]
        ),
    )
    shifted_result = Analysis.from_unit_summary(
        shifted, unit="unit", group="arm", control="c", metrics=[metric]
    ).run()
    assert shifted_result.source != result.source
    assert shifted_result[0].source_snapshot_id != row.source_snapshot_id

    with_sibling = frame.append_column("sibling", pa.array([float(i) for i in range(32)]))
    sibling_rows = Analysis.from_unit_summary(
        with_sibling,
        unit="unit",
        group="arm",
        control="c",
        metrics=[metric, MetricSpec(name="sibling")],
    ).run()
    revenue_row = next(item for item in sibling_rows if item.metric == "aov")
    assert isinstance(revenue_row, LiftEstimate)
    assert revenue_row.require_lift().value == pytest.approx(row.require_lift().value)


def test_cuped_linear_level_support_failure_remains_a_cell():
    import pyarrow as pa

    frame = pa.table(
        {
            "unit": [f"{arm}{i}" for arm in ("c", "t") for i in range(8)],
            "arm": ["c"] * 8 + ["t"] * 8,
            "revenue": [float(i + (arm == "t")) for arm in ("c", "t") for i in range(8)],
            "region": ["north"] * 8 + ["north"] * 4 + ["south"] * 4,
        }
    )
    metric = MetricSpec(
        name="revenue",
        decision_method=Method.cuped_multicovariate(
            name="linear",
            covariates=CupedCovariate(name="region", kind="categorical"),
        ),
    )
    result = Analysis.from_unit_summary(
        frame, unit="unit", group="arm", control="c", metrics=[metric]
    ).run()

    assert len(result) == 1
    assert isinstance(result[0], LiftEstimate)
    assert result[0].failure_code == "estimation.cuped_linear.level_support"
    assert result[0].lift is None


def test_aggregate_lift_estimator_requires_per_unit_cuped_evidence():
    from increment.errors import UnsupportedRequestError
    from increment.estimation.engine import Method, estimate_lift
    from increment.semantics.models import MeanMetric

    with pytest.raises(UnsupportedRequestError) as raised:
        estimate_lift(
            [MeanMetric(name="revenue", entity="user", fact="revenue")],
            [],
            "control",
            methods=[
                Method.cuped_multicovariate(
                    name="linear",
                    covariates=CupedCovariate(name="prior", kind="continuous"),
                )
            ],
        )

    assert raised.value.code == "estimation.engine.cuped_linear_unit_evidence"


def test_definition_covariate_requires_exactly_one_binding():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(name="pre_x", kind="continuous")

    assert raised.value.code == "definition.experiment.covariate_binding_exactly_one"


def test_property_covariate_cannot_also_declare_measure_or_measure_source():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(
            name="pre_x",
            kind="continuous",
            property="tenure",
            measure=Measure(fact="purchase"),
        )

    assert raised.value.code == "definition.experiment.covariate_binding_exactly_one"


def test_measure_source_qualifier_is_rejected():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(
            name="pre_x", kind="continuous", source="events", measure=Measure(fact="purchase")
        )

    assert raised.value.code == "definition.experiment.covariate_source_requires_property"


def test_definition_method_rejects_duplicate_covariate_names():
    x = CupedCovariateBinding(name="pre_x", kind="continuous", property="tenure")
    with pytest.raises(DefinitionError) as raised:
        MethodSpec(
            name="linear",
            variance_reduction="cuped_multicovariate",
            covariates=(x, x),
        )

    assert raised.value.code == "definition.method.covariates"


def test_definition_covariate_refusal_roundtrips_through_copy_and_pickle():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(name="pre_x", kind="continuous")
    error = raised.value

    for restored in (copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
        assert restored.code == error.code
        assert restored.context == error.context


def test_unsupported_kind_is_rejected_by_the_declaration_schema():
    with pytest.raises(ValidationError):
        CupedCovariate.model_validate({"name": "x", "kind": "ordinal"})


def test_scalar_cuped_can_share_a_declared_covariate_with_cuped_linear():
    import pandas as pd

    rows = pd.DataFrame(
        [
            {
                "unit_id": f"{arm[0]}{index}",
                "group_id": arm,
                "revenue": 10.0 + index + 1.5 * (arm == "treatment") + 0.4 * index,
                "pre_revenue": float(index),
            }
            for arm in ("control", "treatment")
            for index in range(60)
        ]
    )
    metric = MetricSpec(
        name="revenue",
        decision_method=Method.cuped(
            covariates=CupedCovariate(name="pre_revenue", kind="continuous")
        ),
        sensitivity_methods=(
            Method.cuped_multicovariate(
                name="linear",
                covariates=CupedCovariate(name="pre_revenue", kind="continuous"),
            ),
        ),
    )
    analysis = Analysis.from_unit_summary(
        rows,
        unit="unit_id",
        group="group_id",
        control="control",
        metrics=(metric,),
    )
    try:
        result = analysis.run()
        estimates = [row for row in result if isinstance(row, LiftEstimate)]
        assert {row.method for row in estimates} == {"cuped", "linear"}
        assert all(row.lift is not None and row.sampling_available for row in estimates)
    finally:
        analysis.close()


def test_missing_sensitivity_covariate_does_not_change_decision_roster():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(17)
    n_per_arm = 40
    x = np.tile(np.arange(n_per_arm, dtype=float), 2)
    arms = ["control"] * n_per_arm + ["treatment"] * n_per_arm
    frame = pd.DataFrame(
        {
            "unit": range(2 * n_per_arm),
            "arm": arms,
            "outcome": 5.0 + 0.5 * x + rng.normal(size=2 * n_per_arm),
            "decision_covariate": x,
            "sensitivity_covariate": [None] * (2 * n_per_arm),
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name="outcome",
                decision_method=Method.cuped(
                    name="decision_cuped",
                    covariates=CupedCovariate(name="decision_covariate", kind="continuous"),
                ),
                sensitivity_methods=(
                    Method.cuped_multicovariate(
                        name="sensitivity_mixed",
                        covariates=CupedCovariate(
                            name="sensitivity_covariate",
                            kind="continuous",
                            missing="refuse",
                        ),
                    ),
                ),
            )
        ],
    )
    rows = analysis.run()

    decision = [row for row in rows if row.method == "decision_cuped"]
    sensitivity = [row for row in rows if row.method == "sensitivity_mixed"]
    assert len(decision) == 1 and isinstance(decision[0], LiftEstimate)
    decision_row = decision[0]
    assert decision_row.group_id == "treatment"
    assert decision_row.sampling_available
    assert decision_row.failure_code is None
    assert len(sensitivity) == 1 and isinstance(sensitivity[0], LiftEstimate)
    sensitivity_row = sensitivity[0]
    assert sensitivity_row.group_id == "treatment"
    assert sensitivity_row.failure_code == "estimation.cuped_linear.missing_refused"


def test_conflicting_cross_role_method_label_is_refused_at_declaration():
    decision = Method.cuped(
        name="shared",
        covariates=CupedCovariate(name="decision_covariate", kind="continuous"),
    )
    sensitivity = Method.cuped_multicovariate(
        name="shared",
        covariates=CupedCovariate(name="sensitivity_covariate", kind="continuous"),
    )

    with pytest.raises(InvalidRequestError) as raised:
        MetricSpec(
            name="outcome",
            decision_method=decision,
            sensitivity_methods=(sensitivity,),
        )

    assert raised.value.code == "estimation.engine.method_names_unique"


def test_method_override_replaces_acquired_scalar_covariate_and_clears_sensitivities():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(41)
    n = 40
    frame = pd.DataFrame(
        {
            "unit": range(2 * n),
            "arm": ["control"] * n + ["treatment"] * n,
            "outcome": 10 + rng.normal(size=2 * n),
            "history_a": rng.normal(size=2 * n),
            "history_b": rng.normal(size=2 * n),
        }
    )
    declared = Method.cuped(
        name="history_a",
        covariates=CupedCovariate(name="history_a", kind="continuous"),
    )
    override = Method.cuped(
        name="history_b",
        covariates=CupedCovariate(name="history_b", kind="continuous"),
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name="outcome",
                decision_method=declared,
                sensitivity_methods=(
                    Method.cuped(
                        name="sensitivity",
                        covariates=CupedCovariate(name="history_a", kind="continuous"),
                    ),
                ),
            )
        ],
    )
    overridden = analysis.run(decision_method=override, sensitivity_methods=())[0]
    reference = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="outcome", decision_method=override)],
    ).run()[0]

    assert isinstance(overridden, LiftEstimate) and isinstance(reference, LiftEstimate)
    assert overridden.method == "history_b"
    assert overridden.require_lift().value == pytest.approx(reference.require_lift().value)
    assert overridden.require_lift().log_se == pytest.approx(reference.require_lift().log_se)


def test_scalar_method_subset_preserves_aliases_that_collide_with_scratch_columns():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(71)
    n = 40
    frame = pd.DataFrame(
        {
            "unit": range(2 * n),
            "arm": ["control"] * n + ["treatment"] * n,
            "outcome": 10 + rng.normal(size=2 * n),
            "y": rng.normal(size=2 * n),
            "__covariate_y": rng.normal(size=2 * n),
        }
    )
    decision = Method.cuped(
        name="y",
        covariates=CupedCovariate(name="y", kind="continuous"),
    )
    sensitivity = Method.cuped(
        name="__covariate_y",
        covariates=CupedCovariate(name="__covariate_y", kind="continuous"),
    )
    combined = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name="outcome",
                decision_method=decision,
                sensitivity_methods=(sensitivity,),
            )
        ],
    ).run()
    reference = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="outcome", decision_method=sensitivity)],
    ).run()[0]
    selected = next(
        row for row in combined if isinstance(row, LiftEstimate) and row.method == sensitivity.name
    )
    assert isinstance(reference, LiftEstimate)

    assert selected.require_lift().value == pytest.approx(reference.require_lift().value)
    assert selected.require_lift().log_se == pytest.approx(reference.require_lift().log_se)


def test_two_mixed_methods_preserve_aliases_colliding_with_scratch_columns():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(71)
    n = 40
    frame = pd.DataFrame(
        {
            "unit": range(2 * n),
            "arm": ["control"] * n + ["treatment"] * n,
            "outcome": 10 + rng.normal(size=2 * n),
            "y": rng.normal(size=2 * n),
            "__covariate_y": rng.normal(size=2 * n),
        }
    )
    first = Method.cuped_multicovariate(
        name="linear_y", covariates=CupedCovariate(name="y", kind="continuous")
    )
    second = Method.cuped_multicovariate(
        name="linear_scratch",
        covariates=CupedCovariate(name="__covariate_y", kind="continuous"),
    )

    def run(methods):
        return Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(
                    name="outcome",
                    decision_method=methods[0],
                    sensitivity_methods=tuple(methods[1:]),
                )
            ],
        ).run()

    together = run((first, second))
    reference = run((first,))
    combined = next(
        row for row in together if isinstance(row, LiftEstimate) and row.method == first.name
    )
    alone = next(row for row in reference if isinstance(row, LiftEstimate))
    assert combined.require_lift().value == pytest.approx(alone.require_lift().value)
    assert combined.require_lift().log_se == pytest.approx(alone.require_lift().log_se)


def test_scalar_sensitivity_uses_its_own_covariate_when_mixed_cuped_is_present():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(19)
    n = 80
    frame = pd.DataFrame(
        {
            "unit": range(2 * n),
            "arm": ["control"] * n + ["treatment"] * n,
            "outcome": 10 + rng.normal(size=2 * n),
            "history_a": rng.normal(size=2 * n),
            "history_b": rng.normal(size=2 * n),
            "mixed_x": rng.normal(size=2 * n),
        }
    )
    first = Method.cuped(
        name="scalar_a",
        covariates=CupedCovariate(name="history_a", kind="continuous"),
    )
    second = Method.cuped(
        name="scalar_b",
        covariates=CupedCovariate(name="history_b", kind="continuous"),
    )
    mixed = Method.cuped_multicovariate(
        name="mixed",
        covariates=CupedCovariate(name="mixed_x", kind="continuous"),
    )

    def run(methods):
        return Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(
                    name="outcome",
                    decision_method=methods[0],
                    sensitivity_methods=tuple(methods[1:]),
                )
            ],
        ).run()

    together = run((first, second, mixed))
    standalone = run((second,))
    result = next(
        row for row in together if isinstance(row, LiftEstimate) and row.method == second.name
    )
    reference = next(row for row in standalone if isinstance(row, LiftEstimate))
    assert result.require_lift().value == pytest.approx(reference.require_lift().value)
    assert result.require_lift().log_se == pytest.approx(reference.require_lift().log_se)


def test_multi_family_scalar_export_refuses_before_writing_payload(tmp_path):
    import numpy as np
    import pandas as pd

    from increment.errors import UnsupportedRequestError

    rng = np.random.default_rng(23)
    n_per_arm = 40
    arms = ["control"] * n_per_arm + ["treatment"] * n_per_arm
    frame = pd.DataFrame(
        {
            "unit": range(2 * n_per_arm),
            "arm": arms,
            "outcome": 10 + rng.normal(size=2 * n_per_arm),
            "history_a": rng.normal(size=2 * n_per_arm),
            "history_b": rng.normal(size=2 * n_per_arm),
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name="outcome",
                decision_method=Method.cuped(
                    name="history_a",
                    covariates=CupedCovariate(name="history_a", kind="continuous"),
                ),
                sensitivity_methods=(
                    Method.cuped(
                        name="history_b",
                        covariates=CupedCovariate(name="history_b", kind="continuous"),
                    ),
                ),
            )
        ],
    )
    combined = {row.method: row for row in analysis.run() if isinstance(row, LiftEstimate)}
    for name in ("history_a", "history_b"):
        single = Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(
                    name="outcome",
                    decision_method=Method.cuped(
                        name=name,
                        covariates=CupedCovariate(name=name, kind="continuous"),
                    ),
                )
            ],
        )
        try:
            reference = single.run()[0]
        finally:
            single.close()
        assert isinstance(reference, LiftEstimate)
        assert combined[name].require_lift().value == pytest.approx(reference.require_lift().value)
        assert combined[name].require_lift().log_se == pytest.approx(
            reference.require_lift().log_se
        )

    path = tmp_path / "multi-family.parquet"

    with pytest.raises(UnsupportedRequestError) as raised:
        analysis.export(path)

    assert raised.value.code == "facade.analysis_config.multifamily_scalar_export"
    assert not path.exists()
