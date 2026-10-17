from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from increment import Analysis, CupedCovariate, Method, MetricSpec
from increment.errors import UnsupportedRequestError
from increment.estimation.results import LiftEstimate


def test_scalar_method_evidence_preserves_each_requested_family():
    rng = np.random.default_rng(417)
    n = 64
    a = rng.normal(size=2 * n)
    b = rng.normal(size=2 * n)
    frame = pd.DataFrame(
        {
            "unit": range(2 * n),
            "arm": ["control"] * n + ["treatment"] * n,
            "outcome": 20 + 2 * a - b + rng.normal(size=2 * n),
            "a": a,
            "b": b,
        }
    )
    methods = tuple(
        Method.cuped(name=name, covariates=CupedCovariate(name=name, kind="continuous"))
        for name in ("a", "b")
    )

    def run(selected):
        with Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(
                    name="outcome",
                    decision_method=selected[0],
                    sensitivity_methods=selected[1:],
                )
            ],
        ) as analysis:
            return {row.method: row for row in analysis.run() if isinstance(row, LiftEstimate)}

    joint = run(methods)
    for method in methods:
        alone = run((method,))[method.name]
        assert joint[method.name].require_lift().value == pytest.approx(
            alone.require_lift().value, rel=1e-10
        )
        assert joint[method.name].require_lift().log_se == pytest.approx(
            alone.require_lift().log_se, rel=1e-10
        )


def test_source_without_method_evidence_refuses_before_reading():
    from increment.frame import from_unit_summary
    from increment.readouts import run

    source = from_unit_summary(
        pd.DataFrame(
            {
                "unit": range(8),
                "arm": ["control"] * 4 + ["treatment"] * 4,
                "outcome": [10.0, 12.0, 11.0, 14.0, 15.0, 12.0, 17.0, 14.0],
                "prior": [1.0, 2.0, 3.0, 4.0] * 2,
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name="outcome",
                decision_method=Method.cuped(
                    covariates=CupedCovariate(name="prior", kind="continuous")
                ),
            )
        ],
    )

    class AggregateOnlySource:
        def __getattr__(self, name):
            if name == "moments_for_method":
                raise AttributeError(name)
            return getattr(source, name)

        def moments(self, *args, **kwargs):
            pytest.fail("unsupported route read aggregate rows")

        def unit_frame(self, *args, **kwargs):
            pytest.fail("unsupported route read unit rows")

        def assignment_counts(self, *args, **kwargs):
            pytest.fail("unsupported route read assignment counts")

        def unit_counts(self, *args, **kwargs):
            pytest.fail("unsupported route read unit counts")

    with pytest.raises(UnsupportedRequestError) as raised:
        run(cast(Any, AggregateOnlySource()))
    assert raised.value.code == "readout.scalar_method_evidence"
    assert raised.value.context["route"] == "AggregateOnlySource"


def _two_family_frame():
    rng = np.random.default_rng(281)
    n = 160
    a, b = rng.normal(size=(2, 2 * n))
    treated = np.repeat([0, 1], n)
    return pd.DataFrame(
        {
            "unit": range(2 * n),
            "arm": np.where(treated, "treatment", "control"),
            "day": 1,
            "outcome": 20 + 2 * a - b + 5 * treated + rng.normal(size=2 * n),
            "a": a,
            "b": b,
            "uptake": treated * rng.binomial(1, 0.75, size=2 * n),
        }
    )


def _scalar(name, *, missing="refuse"):
    return Method.cuped(
        name=name, covariates=CupedCovariate(name=name, kind="continuous", missing=missing)
    )


@pytest.mark.parametrize("panel", [False, True])
@pytest.mark.parametrize("secondary", [False, True])
def test_encouragement_scalar_families_match_standalone(panel, secondary):
    from increment import AnalysisPlan
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    frame = _two_family_frame()
    methods = (_scalar("a"), _scalar("b"))
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="uptake"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment affects outcomes through uptake"
        ),
    )

    def run(selected):
        constructor = Analysis.from_unit_panel if panel else Analysis.from_unit_summary
        with constructor(
            frame,
            unit="unit",
            group="arm",
            metrics=[
                MetricSpec(
                    name="outcome",
                    decision_method=selected[0],
                    sensitivity_methods=selected[1:],
                )
            ],
            design=design,
            uptake="uptake",
            plan=(
                AnalysisPlan(secondaries=["outcome"], q=0.1)
                if secondary
                else AnalysisPlan(primary="outcome")
            ),
            **({"date": "day"} if panel else {}),
        ) as analysis:
            return {
                (row.method, row.estimand): row
                for row in analysis.run(estimands=("itt", "late"))
                if isinstance(row, LiftEstimate)
            }

    joint = run(methods)
    for method in methods:
        alone = run((method,))
        for estimand in ("itt", "late"):
            actual = joint[method.name, estimand]
            expected = alone[method.name, estimand]
            assert actual.require_lift().value == pytest.approx(
                expected.require_lift().value, rel=1e-10
            )
            assert actual.require_lift().log_se is not None
            assert actual.require_lift().log_se == pytest.approx(
                expected.require_lift().log_se, rel=1e-10
            )
            if estimand == "itt":
                assert actual.abs_diff == pytest.approx(expected.abs_diff, rel=1e-10)
                assert actual.abs_se == pytest.approx(expected.abs_se, rel=1e-10)


@pytest.mark.parametrize("mixed", [False, True])
def test_selected_scalar_secondary_with_override_matches_standalone(mixed):
    from increment import AnalysisPlan

    frame = _two_family_frame()
    decision = _scalar("b")
    companions = [_scalar("a")]
    if mixed:
        companions.append(
            Method.cuped_multicovariate(
                name="mixed", covariates=CupedCovariate(name="a", kind="continuous")
            )
        )

    def run(joint):
        with Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            control="control",
            metrics=[MetricSpec(name="outcome", decision_method=_scalar("a"))],
            plan=AnalysisPlan(secondaries=["outcome"], q=0.1),
        ) as analysis:
            return {
                row.method: row
                for row in analysis.run(
                    decision_method=decision,
                    sensitivity_methods=tuple(companions) if joint else (),
                )
                if isinstance(row, LiftEstimate)
            }

    actual = run(True)["b"]
    expected = run(False)["b"]
    assert actual.discovery is True and expected.discovery is True
    assert actual.require_lift().value == pytest.approx(expected.require_lift().value, rel=1e-10)
    assert actual.require_lift().log_se == pytest.approx(expected.require_lift().log_se, rel=1e-10)
    assert actual.require_lift().lb == pytest.approx(expected.require_lift().lb, rel=1e-10)
    assert actual.require_lift().ub == pytest.approx(expected.require_lift().ub, rel=1e-10)


@pytest.mark.parametrize("panel", [False, True])
@pytest.mark.parametrize("role", ["decision", "sensitivity"])
@pytest.mark.parametrize("missing", ["refuse", "impute"])
def test_encouragement_scalar_override_rechecks_missing_policy(panel, role, missing):
    from contextlib import nullcontext

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    frame = _two_family_frame()
    if missing == "refuse":
        frame.loc[0, "a"] = np.nan
    else:
        frame["a"] = np.nan
    constructor = Analysis.from_unit_panel if panel else Analysis.from_unit_summary
    from increment.errors import IncrementWarning

    capture = pytest.warns(IncrementWarning) if missing == "refuse" else nullcontext()
    with capture as caught:
        analysis = constructor(
            frame,
            unit="unit",
            group="arm",
            metrics=[MetricSpec(name="outcome", decision_method=_scalar("a", missing="impute"))],
            design=Encouragement(
                control_group="control",
                uptake=UptakeSpec(fact="uptake"),
                exclusion_restriction=ExclusionRestriction(
                    acknowledged=True, justification="assignment affects outcomes through uptake"
                ),
            ),
            uptake="uptake",
            **({"date": "day"} if panel else {}),
        )
    if missing == "refuse":
        assert caught is not None
        assert {
            warning.message.code
            for warning in caught
            if isinstance(warning.message, IncrementWarning)
        } == {"frame.validation.metric_covariate_missing_imputed"}
    try:
        bad, good, naive = _scalar("a", missing=missing), _scalar("b"), Method.unadjusted()
        methods = (bad, naive, good) if role == "decision" else (naive, bad, good)
        rows = analysis.run(
            decision_method=methods[0], sensitivity_methods=methods[1:], estimands=("itt", "late")
        )
        failed = [row for row in rows if isinstance(row, LiftEstimate) and row.method == bad.name]
        assert {row.estimand for row in failed} == {"itt", "late"}
        assert all(row.lift is None for row in failed)
        expected_code = (
            "frame.validation.metric_covariate_column"
            if missing == "refuse"
            else "frame.validation.metric_covariate_impute_all_null"
        )
        assert {row.failure_code for row in failed} == {expected_code}
        assert {row.method_role for row in failed} == {role}
        valid = [
            row
            for row in rows
            if isinstance(row, LiftEstimate) and row.method in (good.name, naive.name)
        ]
        assert {(row.method, row.estimand) for row in valid} == {
            (method.name, estimand) for method in (good, naive) for estimand in ("itt", "late")
        }
        assert all(row.lift is not None for row in valid)
    finally:
        analysis.close()


@pytest.mark.parametrize("panel", [False, True])
def test_encouragement_scalar_families_share_one_weak_stage_diagnostic(panel):
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    frame = _two_family_frame()
    frame["uptake"] = 0
    methods = (_scalar("a"), _scalar("b"))
    constructor = Analysis.from_unit_panel if panel else Analysis.from_unit_summary
    with constructor(
        frame,
        unit="unit",
        group="arm",
        metrics=[
            MetricSpec(name="outcome", decision_method=methods[0], sensitivity_methods=methods[1:])
        ],
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="uptake"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment affects outcomes through uptake"
            ),
        ),
        uptake="uptake",
        **({"date": "day"} if panel else {}),
    ) as analysis:
        results = analysis.run(estimands=("itt", "late"))
    itt = [row for row in results if isinstance(row, LiftEstimate) and row.estimand == "itt"]
    late = [row for row in results if isinstance(row, LiftEstimate) and row.estimand == "late"]
    diagnostics = [
        row for row in results if isinstance(row, LiftEstimate) and row.estimand == "compliance"
    ]
    assert {row.method for row in itt} == {"a", "b"}
    assert all(row.lift is not None for row in itt)
    assert {row.method for row in late} == {"a", "b"}
    assert all(row.lift is None for row in late)
    assert len(diagnostics) == 1
    assert diagnostics[0].metric == "outcome"


def test_encouragement_refuses_multicovariate_sensitivity():
    from increment.errors import UnsupportedRequestError
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    with Analysis.from_unit_summary(
        _two_family_frame(),
        unit="unit",
        group="arm",
        metrics=[MetricSpec(name="outcome", decision_method=_scalar("a"))],
        uptake="uptake",
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="uptake"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment affects outcomes through uptake"
            ),
        ),
    ) as analysis:
        with pytest.raises(UnsupportedRequestError) as raised:
            analysis.run(
                sensitivity_methods=(
                    Method.cuped_multicovariate(
                        name="mixed",
                        covariates=CupedCovariate(name="b", kind="continuous"),
                    ),
                )
            )
    assert raised.value.code == "estimation.engine.cuped_linear_encouragement"
