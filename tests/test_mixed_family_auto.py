"""Automatic composition: continuous or binary ITT plus the design's Bernoulli
uptake compliance cell, bound from metadata on every ingress path."""

import datetime as dt
from fractions import Fraction

import pytest

from increment import Analysis
from increment.errors import CapabilityError
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec, synthesise_metric
from increment.plan import bind_automatic_sequential_plan
from increment.semantics.design import Encouragement, ExclusionRestriction, Randomized, UptakeSpec
from increment.semantics.models import AnalysisPlan, InferenceSpec
from increment.semantics.sequential import SequentialCompliancePolicy
from increment.sequential_source import frame_observation_mapping

DESIGN = Encouragement(
    control_group="control",
    uptake=UptakeSpec(fact="clicked"),
    exclusion_restriction=ExclusionRestriction(
        acknowledged=True, justification="the prompt affects revenue only through clicks"
    ),
    allocation={"control": 0.5, "treatment": 0.5},
)
SPECS = [MetricSpec(name="revenue", type="mean"), MetricSpec(name="orders", type="mean")]


def _bound(
    *, kind="asymptotic_mean", specs=SPECS, compliance, secondaries=(), design=DESIGN, q=0.1
):
    primary = specs[0].name
    plan = AnalysisPlan(
        primary=primary,
        secondaries=list(secondaries) or None,
        q=q,
        inference=InferenceSpec(kind=kind),
        compliance=compliance,
    )
    return bind_automatic_sequential_plan(
        plan,
        [synthesise_metric(s) for s in specs],
        design=design,
        source_id="frame",
        source_mapping=frame_observation_mapping(unit="unit", group="arm", uptake="clicked"),
        transformations=specs,
        path="frame",
    )


def test_automatic_scalar_mean_composes_bernoulli_uptake():
    policy = SequentialCompliancePolicy(alpha=Fraction(1, 20))
    registration = _bound(compliance=policy).inference.registration
    laws = {m.metric: (m.law, m.observable) for m in registration.models}
    assert laws == {
        "orders": ("scalar_mean", "outcome"),
        "revenue": ("scalar_mean", "outcome"),
        "uptake": ("bernoulli", "uptake"),
    }
    compliance = next(c for c in registration.roster if c.estimand == "compliance")
    assert compliance.alpha == policy.alpha
    assert compliance.family is False


def test_compliance_in_family_shares_the_equal_bonferroni_split():
    policy = SequentialCompliancePolicy(alpha=Fraction(1, 20), family=True)
    registration = _bound(compliance=policy, secondaries=["orders"]).inference.registration
    family = [c for c in registration.roster if c.family]
    assert {c.metric for c in family} == {"orders", "uptake"}
    assert all(c.alpha <= registration.q / len(family) for c in family)
    assert sum(c.alpha for c in family) <= registration.q


def test_automatic_scalar_mean_without_compliance_is_itt_alone():
    """No compliance policy: exactly S0-1's capability, unaffected by this task."""
    registration = _bound(compliance=None).inference.registration
    assert {m.law for m in registration.models} == {"scalar_mean"}


def test_automatic_bernoulli_composes_bernoulli_uptake():
    """auto_register_bernoulli's own mechanism fix: binary ITT (all-exact
    family, no mixing needed) plus uptake compliance under encouragement."""
    specs = [MetricSpec(name="converted", type="conversion")]
    policy = SequentialCompliancePolicy(alpha=Fraction(1, 20))
    registration = _bound(
        kind="always_valid", specs=specs, compliance=policy
    ).inference.registration
    assert {m.metric: m.observable for m in registration.models} == {
        "converted": "outcome",
        "uptake": "uptake",
    }
    assert all(m.law == "bernoulli" for m in registration.models)


def test_automatic_bernoulli_under_encouragement_no_longer_refuses():
    """Regression: this call raised sequential.route.unsupported before the fix."""
    specs = [MetricSpec(name="converted", type="conversion")]
    registration = _bound(kind="always_valid", specs=specs, compliance=None).inference.registration
    assert registration.models[0].law == "bernoulli"


def test_compliance_under_a_randomized_design_refuses_by_code():
    with pytest.raises(CapabilityError) as raised:
        _bound(
            compliance=SequentialCompliancePolicy(alpha=Fraction(1, 20)),
            design=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
        )
    assert raised.value.code == "sequential.source.invalid"


def test_compliance_alpha_binds_a_yaml_decimal_exactly():
    """0.05, its string, and Fraction(1, 20) must all bind the same Fraction --
    otherwise a YAML user's ordinary decimal produces a different registration
    than a Python caller's exact Fraction."""
    exact = Fraction(1, 20)
    for spelling in (0.05, "0.05", exact):
        policy = SequentialCompliancePolicy(alpha=spelling)
        assert policy.alpha == exact
    registrations = [
        _bound(compliance=SequentialCompliancePolicy(alpha=spelling)).inference.registration
        for spelling in (0.05, "0.05", exact)
    ]
    assert len({r.model_dump_json() for r in registrations}) == 1


def _frame(n=120):
    import pandas as pd

    rows = []
    for i in range(n):
        for arm in ("control", "treatment"):
            clicked = int(i % 4 != 0) if arm == "treatment" else int(i % 4 == 3)
            rows.append(
                {
                    "unit": f"{i:05d}-{arm}",
                    "arm": arm,
                    "revenue": float(
                        (6, 10, 14, 18)[i % 4] if arm == "treatment" else (1, 2, 3, 4)[i % 4]
                    ),
                    "clicked": clicked,
                    "orders": float(i % 3 + (1 if arm == "treatment" else 0)),
                    "exposure_date": i,
                }
            )
    return pd.DataFrame(rows)


def _summary():
    return Analysis.from_unit_summary(
        _frame(),
        unit="unit",
        group="arm",
        metrics=SPECS,
        experiment_id="frame",
        design=DESIGN,
        uptake="clicked",
        exposure_date="exposure_date",
        plan=AnalysisPlan(
            primary="revenue",
            inference=InferenceSpec(kind="asymptotic_mean"),
            compliance=SequentialCompliancePolicy(alpha=Fraction(1, 20)),
        ),
    )


def test_unit_summary_runs_the_composition_end_to_end():
    analysis = _summary()
    rows = analysis.run(estimands=("itt", "compliance"))
    assert all(isinstance(row, LiftEstimate) for row in rows)
    by_estimand = {(row.metric, row.estimand): row for row in rows if isinstance(row, LiftEstimate)}
    assert set(by_estimand) == {("orders", "itt"), ("revenue", "itt"), ("uptake", "compliance")}
    uptake_result = by_estimand["uptake", "compliance"].sequential_result
    revenue_result = by_estimand["revenue", "itt"].sequential_result
    assert uptake_result is not None and uptake_result.checkpoint.model.law == "bernoulli"
    assert revenue_result is not None and revenue_result.checkpoint.model.law == "scalar_mean"


def test_compiled_plan_round_trips_the_wire():
    from typing import cast

    from increment.decision_wire import compiled_plan_from_dict, compiled_plan_to_dict
    from increment.plan import compile_decision_plan

    compiled = compile_decision_plan(
        _bound(compliance=SequentialCompliancePolicy(alpha=Fraction(1, 20))),
        [synthesise_metric(s) for s in SPECS],
        path="frame",
        design=DESIGN,
        estimands=("itt", "compliance"),
    )
    payload = compiled_plan_to_dict(compiled)
    assert payload["wire_version"] == 3
    inference_payload = cast("dict[str, object]", payload["inference"])
    assert inference_payload["kind"] == "asymptotic_mean"
    registration = cast("dict[str, object]", inference_payload["registration"])
    models = cast("list[dict[str, str]]", registration["models"])
    assert {model["observable"] for model in models} == {"outcome", "uptake"}
    restored = compiled_plan_from_dict(payload)
    assert restored.inference == compiled.inference


def test_from_moments_replays_the_composed_checkpoint_exactly(tmp_path):
    import pyarrow.parquet as pq

    summary = _summary()
    snapshot = summary.capture_sequential(finalized=True)
    oracle_rows = summary.run(estimands=("itt", "compliance"))
    assert all(isinstance(row, LiftEstimate) for row in oracle_rows)
    oracle = {
        (row.metric, row.estimand): row for row in oracle_rows if isinstance(row, LiftEstimate)
    }
    path = tmp_path / "checkpoint.parquet"
    summary.export(path)
    replay = Analysis.from_moments(pq.read_table(path).to_pylist(), metrics=SPECS, design=DESIGN)
    assert replay.sequential_snapshot() == snapshot
    replayed_rows = replay.run(estimands=("itt", "compliance"))
    assert all(isinstance(row, LiftEstimate) for row in replayed_rows)
    replayed = {
        (row.metric, row.estimand): row for row in replayed_rows if isinstance(row, LiftEstimate)
    }
    assert {k: r.require_sequential_result() for k, r in replayed.items()} == {
        k: r.require_sequential_result() for k, r in oracle.items()
    }


_PANEL_DESIGN = Encouragement(
    control_group="control",
    uptake=UptakeSpec(fact="clicked", window_days=1),
    exclusion_restriction=ExclusionRestriction(
        acknowledged=True, justification="the prompt affects revenue only through clicks"
    ),
    allocation={"control": 0.5, "treatment": 0.5},
)
_PANEL_SPECS = [
    MetricSpec(name="revenue", type="mean", window_days=1),
    MetricSpec(name="orders", type="mean", window_days=1),
]


def _panel():
    import pandas as pd

    day = dt.date(2024, 1, 1)
    rows = [{**row, "day": day, "exposed": day} for row in _frame().to_dict("records")]
    return pd.DataFrame(rows)


def test_unit_panel_runs_the_composition_end_to_end():
    analysis = Analysis.from_unit_panel(
        _panel(),
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
        metrics=_PANEL_SPECS,
        experiment_id="frame",
        design=_PANEL_DESIGN,
        uptake="clicked",
        observation_end=dt.date(2024, 1, 1),
        plan=AnalysisPlan(
            primary="revenue",
            inference=InferenceSpec(kind="asymptotic_mean"),
            compliance=SequentialCompliancePolicy(alpha=Fraction(1, 20)),
        ),
    )
    analysis.capture_sequential(finalized=True, as_of=dt.date(2024, 1, 1))
    rows = analysis.run(estimands=("itt", "compliance"))
    assert all(isinstance(row, LiftEstimate) for row in rows)
    by_estimand = {(row.metric, row.estimand): row for row in rows if isinstance(row, LiftEstimate)}
    assert set(by_estimand) == {("orders", "itt"), ("revenue", "itt"), ("uptake", "compliance")}
    uptake_result = by_estimand["uptake", "compliance"].sequential_result
    assert uptake_result is not None and uptake_result.checkpoint.model.law == "bernoulli"


# The parity case `sequential_composed_itt_and_uptake` (tests/parity_harness/cases.py)
# covers this composition on every constructor except from_switchback_panel, which
# has no sequential switchback construction.
