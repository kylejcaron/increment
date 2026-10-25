from __future__ import annotations

from typing import Any, cast

import pyarrow as pa
import pytest

from increment import Analysis, AnalysisPlan, Method, MetricSpec
from increment.breakout.estimates import BreakoutEstimate
from increment.errors import IncrementWarning
from increment.estimation.decision_types import PValueEvidence
from increment.estimation.inference import Normal
from increment.estimation.results import LiftEstimate
from increment.evidence import EvidenceSource
from increment.frame import CupedCovariate
from increment.semantics.models import ExperimentMetric


def _rows(*, unsupported: bool = False, nonpositive: bool = False) -> pa.Table:
    rows: list[dict[str, Any]] = []
    for group in ("control", "treatment"):
        for index in range(24):
            pre = (
                "treatment-only"
                if unsupported and group == "treatment" and index == 0
                else ("a", "b", "c")[index % 3]
            )
            outcome = (10.0 if group == "control" else (-2.0 if nonpositive else 12.0)) + index % 3
            rows.append(
                {
                    "unit": f"{group}-{index}",
                    "arm": group,
                    "outcome": outcome,
                    "pre": pre,
                }
            )
    return pa.Table.from_pylist(rows)


def _metric(method: Method) -> MetricSpec:
    return MetricSpec(
        name="outcome",
        preferred_direction="decrease",
        decision_method=method,
    )


def _analysis(table: pa.Table, *, plan: AnalysisPlan | None = None) -> Analysis:
    return Analysis.from_unit_summary(
        table,
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            _metric(
                Method.cuped_multicovariate(
                    name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
                )
            )
        ],
        plan=plan,
    )


def test_failed_linear_sensitivity_does_not_break_ordinary_secondary_fcr() -> None:
    analysis = _analysis(
        _rows(unsupported=True),
        plan=AnalysisPlan(q=0.2, secondaries=["outcome"]),
    )
    with pytest.warns(IncrementWarning) as caught:
        result = cast(
            list[LiftEstimate],
            list(
                analysis.run(
                    decision_method=Method(name="plain"),
                    sensitivity_methods=[
                        Method.cuped_multicovariate(
                            name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
                        )
                    ],
                )
            ),
        )
    assert any(
        getattr(warning.message, "code", None) == "analysis.run.override_replaces_declared"
        for warning in caught
    )
    by_method = {row.method: row for row in result}
    assert by_method["plain"].lift is not None
    assert by_method["linear"].lift is None
    assert by_method["linear"].failure_code == "estimation.cuped_linear.level_support"


def test_fit_wide_failure_retains_direction_margin_and_posterior_reason() -> None:
    analysis = _analysis(
        _rows(unsupported=True),
        plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="outcome", margin_abs=0.2)]),
    )
    (row,) = cast(list[LiftEstimate], list(analysis.run(prior=Normal(mu=0.0, sigma=0.1))))

    assert row.failure_code == "estimation.cuped_linear.level_support"
    assert row.preferred_direction == "decrease"
    assert row.null_abs == pytest.approx(0.2)
    assert row.posterior_available is False
    assert row.posterior_reason_code == "estimation.results.lift.t_posterior_sufficient_statistics"


def test_linear_methods_with_same_construction_share_one_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import increment.estimation.cuped.multicovariate as cuped_multicovariate

    original = cuped_multicovariate.fit_cuped_multicovariate
    calls = 0

    def counted(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cuped_multicovariate, "fit_cuped_multicovariate", counted)
    analysis = _analysis(_rows())
    methods = [
        Method.cuped_multicovariate(
            name="linear-decision", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
        Method.cuped_multicovariate(
            name="linear-sensitivity", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
    ]

    with pytest.warns(IncrementWarning) as caught:
        result = cast(
            list[LiftEstimate],
            list(analysis.run(decision_method=methods[0], sensitivity_methods=[methods[1]])),
        )
    assert any(
        getattr(warning.message, "code", None) == "analysis.run.override_replaces_declared"
        for warning in caught
    )

    assert {row.method for row in result} == {method.name for method in methods}
    assert calls == 1


def test_relative_projection_failure_keeps_absolute_guardrail_inference() -> None:
    analysis = _analysis(
        _rows(nonpositive=True),
        plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="outcome", margin_abs=0.01)]),
    )
    results = analysis.run(prior=Normal(mu=0.0, sigma=0.1))
    (row,) = cast(list[LiftEstimate], list(results))

    assert row.lift is None
    assert row.abs_diff is not None and row.abs_diff < 0.0
    assert row.abs_se is not None and row.abs_se > 0.0
    assert row.abs_lb is not None and row.abs_ub is not None
    assert row.abs_reference_kind == "t" and row.abs_reference_df is not None
    assert row.relative_unavailable_reason == "nonpositive_arm_mean"
    assert row.failure_code is None
    assert row.relative_error_code == "estimation.cuped_linear.nonpositive_component"
    assert row.relative_error_context == {"family": "outcome", "arm": "treatment", "value": -1.0}
    assert row.p_value() is not None
    assert row.posterior_available is False
    assert row.posterior_reason_code == "estimation.results.lift.t_posterior_sufficient_statistics"
    assert results.metadata is not None
    (record,) = results.metadata.cells
    assert record.sampling is not None and record.sampling.available is True
    assert isinstance(record.sampling.evidence, PValueEvidence)
    assert record.sampling.evidence.p_value == pytest.approx(row.p_value())
    assert record.posterior is not None and record.posterior.available is False
    assert record.posterior.reason_code == row.posterior_reason_code
    from increment.estimation.readout_types import ReadoutResults

    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored.metadata == results.metadata
    restored_row = restored[0]
    assert isinstance(restored_row, LiftEstimate)
    assert restored_row.p_value() == pytest.approx(row.p_value())


@pytest.mark.parametrize("override", ["decision", "sensitivity"])
def test_unavailable_linear_override_refuses_before_evidence_acquisition(override):
    from increment.errors import CodedError
    from increment.readouts import run
    from tests.test_cuped_linear_acceptance import _summary_source

    source = _summary_source(metric=MetricSpec(name="revenue"))
    linear = Method.cuped_multicovariate(
        name="linear",
        covariates=CupedCovariate(name="missing_covariate", kind="continuous"),
    )
    with pytest.raises(CodedError) as raised:
        if override == "decision":
            run(cast(EvidenceSource, source), decision_method=linear)
        else:
            run(cast(EvidenceSource, source), sensitivity_methods=[linear])
    assert raised.value.code == "facade.analysis_config.cuped_linear_covariates"
    assert source.read_calls == 0


def test_available_unadjusted_source_column_supports_mixed_cuped_override():
    rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "revenue": float(index + (group == "treatment") * 3),
            "prior": float(index % 7),
        }
        for group in ("control", "treatment")
        for index in range(40)
    ]
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        control="control",
        metrics=[MetricSpec(name="revenue")],
    )

    result = analysis.run(
        decision_method=Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="prior", kind="continuous")
        )
    )[0]

    assert isinstance(result, LiftEstimate)
    log_se = result.require_lift().log_se
    assert log_se is not None and log_se > 0.0


def test_m1_unavailable_readouts_replay_exact_fit_refusal():
    import copy
    import pickle

    import numpy as np

    from increment.errors import CodedError
    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.estimation.readout_types import ReadoutResults

    arms = np.repeat(["control", "treatment"], 10)
    phase = np.tile(np.arange(10) * 2 * np.pi / 10, 2)
    x = np.column_stack((np.sin(phase), np.cos(phase), np.sin(2 * phase)))
    y = 10 + np.arange(20) / 10
    covariates = {f"x{i}": x[:, i] for i in range(3)}
    kinds = dict.fromkeys(covariates, "continuous")
    with pytest.raises(CodedError) as raised:
        fit_cuped_multicovariate(
            y,
            arms,
            control="control",
            arm_order=("control", "treatment"),
            covariates=covariates,
            kinds=kinds,
        )
    expected = raised.value
    frame = pa.table(
        {
            "unit": [f"u{i}" for i in range(20)],
            "arm": arms,
            "day": [1] * 20,
            "segment": ["all"] * 20,
            "outcome": y,
            **covariates,
        }
    )
    metric = MetricSpec(
        name="outcome",
        decision_method=Method.cuped_multicovariate(
            name="linear",
            covariates=tuple(CupedCovariate(name=name, kind="continuous") for name in covariates),
        ),
    )
    analysis = Analysis.from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics=[metric],
        breakouts=["segment"],
    )
    headline = analysis.run()
    with pytest.warns():
        breakout = analysis.run_breakout()
    for results in (headline, breakout):
        for restored in (
            results,
            copy.deepcopy(results),
            pickle.loads(pickle.dumps(results)),
            ReadoutResults.model_validate_json(results.model_dump_json()),
        ):
            (row,) = restored
            assert isinstance(row, (LiftEstimate, BreakoutEstimate))
            assert row.lift is None and row.sampling_available is False
            assert row.failure_code == expected.code
            assert row.failure_context == expected.context
            assert row.sampling_reason_context == expected.context
            assert restored.metadata is not None
            (record,) = restored.metadata.cells
            assert record.failure is not None and record.failure.context == expected.context
            assert record.sampling is not None and record.sampling.available is False
            with pytest.raises(TypeError):
                cast(dict[str, object], row.failure_context)["region"] = "changed"


def _m1_rejected_frame() -> tuple[pa.Table, MetricSpec, Any]:
    import numpy as np

    from increment.errors import CodedError
    from increment.estimation.cuped import fit_cuped_multicovariate

    arms = np.repeat(["control", "treatment"], 10)
    phase = np.tile(np.arange(10) * 2 * np.pi / 10, 2)
    x = np.column_stack((np.sin(phase), np.cos(phase), np.sin(2 * phase)))
    y = 10 + np.arange(20) / 10
    covariates = {f"x{i}": x[:, i] for i in range(3)}
    with pytest.raises(CodedError) as raised:
        fit_cuped_multicovariate(
            y,
            arms,
            control="control",
            arm_order=("control", "treatment"),
            covariates=covariates,
            kinds=dict.fromkeys(covariates, "continuous"),
        )
    frame = pa.table(
        {"unit": [f"u{i}" for i in range(20)], "arm": arms, "outcome": y, **covariates}
    )
    metric = MetricSpec(
        name="outcome",
        decision_method=Method.cuped_multicovariate(
            name="linear",
            covariates=tuple(CupedCovariate(name=name, kind="continuous") for name in covariates),
        ),
    )
    return frame, metric, raised.value


@pytest.mark.parametrize("companion", ["plain", "linear-sensitivity"])
def test_m1_decision_refusal_survives_companion_sensitivity_method(companion: str) -> None:
    frame, metric, expected = _m1_rejected_frame()
    analysis = Analysis.from_unit_summary(
        frame, unit="unit", group="arm", control="control", metrics=[metric]
    )
    if companion == "plain":
        sensitivity = Method(name="plain")
    else:
        decision_method = metric.decision_method
        assert decision_method is not None
        covariates = decision_method.covariates
        assert covariates is not None
        sensitivity = Method.cuped_multicovariate(name=companion, covariates=covariates)

    results = analysis.run(sensitivity_methods=[sensitivity])

    by_method = {row.method: row for row in results}
    linear = by_method["linear"]
    assert isinstance(linear, LiftEstimate)
    assert linear.failure_code == expected.code
    assert linear.failure_context == expected.context
    assert results.metadata is not None
    decision = next(record for record in results.metadata.cells if record.cell.method == "linear")
    assert decision.failure is not None
    assert decision.failure.code == expected.code
    assert decision.failure.context == expected.context
    assert linear.decision_scope_complete is False


def test_nonfinite_support_observable_is_retained_canonically() -> None:
    from increment.estimation.readout_types import ReadoutResults

    rows = [
        {
            "unit": f"{group}-{index}",
            "arm": group,
            "revenue": float(index + 1),
            "orders": [1.0, -1.0, 1e-320, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][index],
            "pre": ("a", "b")[index % 2],
        }
        for group in ("control", "treatment")
        for index in range(10)
    ]
    metric = MetricSpec(
        name="revenue_per_order",
        type="ratio",
        numerator="revenue",
        denominator="orders",
        decision_method=Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
    )
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(rows), unit="unit", group="arm", control="control", metrics=[metric]
    )

    results = analysis.run()

    (row,) = results
    assert isinstance(row, LiftEstimate)
    assert row.lift is None
    assert row.failure_code == "estimation.cuped_linear.support_region"
    assert row.failure_context is not None
    assert row.failure_context["violations"] == (("ratio_denominator_precision", "inf", 0.15),)
    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    assert restored[0].failure_context == row.failure_context
    assert restored.metadata == results.metadata
