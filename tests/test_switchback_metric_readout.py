"""Retained-window fits keep their effect or their diagnosis through storage."""

from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

from increment import from_switchback_frame, run
from increment.estimation.engine import Method
from increment.frame import MetricSpec
from increment.result_envelope import render
from increment.results import ReadoutResults
from increment.semantics.assignment import (
    IndependentBernoulliOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.design import Randomized
from increment.store import HistorySelector, ParquetResultsStore
from increment.tables import estimates_to_readout
from tests.capability_consumers_scalar import RATIO_DENOMINATOR_ADVISORY


def _experiment(metrics):
    rows = []
    for unit in range(6):
        for period in range(2):
            treated = period == unit % 2
            rows.append(
                {
                    "u": str(unit),
                    "c": 0,
                    "p": period,
                    "s": 0,
                    "arm": "B" if treated else "A",
                    "y": 2 * (unit + 1) if treated else 0,
                    "two": 2,
                    "zero": int(not treated),
                    "rare": int(not treated or unit == 0),
                    "level": 2**55 + ((3 if unit % 2 else -1) if treated else 0),
                    "one": 1,
                }
            )
    forecasts = pl.DataFrame(
        {
            "u": row["u"],
            "c": row["c"],
            "p": row["p"],
            "prior_position": row["p"],
            "same_position": row["p"],
            "constant": 1,
            "forecast_at": date(2025, 1, 1),
            "assigned_at": date(2025, 1, 2),
        }
        for row in rows
    )
    return from_switchback_frame(
        pl.DataFrame(rows),
        unit="u",
        cycle="c",
        period="p",
        step="s",
        group="arm",
        metrics=metrics,
        experiment_id="retained",
        pre_assignment_forecasts=forecasts,
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=0, observation_steps=1),
        ),
        identification=Randomized(control_group="A", allocation={"A": 0.5, "B": 0.5}),
    )


@pytest.mark.integration
def test_unavailable_ratios_keep_their_point_status_and_valid_siblings_in_storage(tmp_path):
    metrics = [
        MetricSpec(name="orders", type="mean", value_column="y"),
        MetricSpec(
            name="adjusted",
            type="mean",
            value_column="y",
            decision_method=Method.cuped(covariates="prior_position"),
        ),
        MetricSpec(
            name="unadjustable",
            type="mean",
            value_column="y",
            decision_method=Method.cuped(covariates="constant"),
        ),
        MetricSpec(name="level_difference", type="ratio", numerator="level", denominator="one"),
        *(
            MetricSpec(name=d, type="ratio", numerator="y", denominator=d)
            for d in ("two", "rare", "zero")
        ),
    ]
    experiment, reader = _experiment(metrics)
    store = ParquetResultsStore(tmp_path)
    with reader:
        saved = run(experiment, reader, results_store=store)
        (alone,) = run(experiment, reader, metrics=["orders"]).decisions.cells
    frame = saved.readout().to_frame(backend="polars")
    assert isinstance(frame, pl.DataFrame)
    rows = {r["metric"]: r for r in frame.to_dicts()}
    assert {metric: r["estimate"] for metric, r in rows.items()} == {
        "orders": 7,
        "adjusted": pytest.approx(7),
        "two": 3.5,
        "rare": 42,
        "zero": None,
        "level_difference": 1,
        "unadjustable": None,
    }
    assert (rows["adjusted"]["method"], rows["adjusted"]["dof"]) == (
        "switchback_cuped_jackknife_t",
        4,
    )
    assert rows["adjusted"]["forecast_column"] == "prior_position"
    assert (rows["two"]["reference"], rows["two"]["dof"]) == ("switchback_ratio_delta_t", 5)
    assert rows["orders"]["sampling_available"] and rows["orders"]["standard_error"] > 0
    assert not rows["orders"]["decision_scope_complete"]
    expected = RATIO_DENOMINATOR_ADVISORY.outcome("switchback-weak-denominator").expected
    assert (rows["rare"]["failure_code"],) == expected.codes

    handle = store.open_run("retained", saved.run.analysis_spec_hash, saved.run.run_id)
    envelope = handle.envelope()
    stored = {cell.identity.key.metric: cell for cell in envelope.cells}
    verdicts = {v.identity_hash: v for v in saved.decisions.cells}
    assert verdicts[stored["orders"].identity.hash()].status == alone.status == "reject"
    projected = {
        json.loads(r["identity"])["key"]["metric"]: r for r in envelope.to_arrow().to_pylist()
    }
    report = render(envelope, audience="full")
    rank = stored["unadjustable"]
    assert rank.summary is not None and rank.summary.point is None
    assert verdicts[rank.identity.hash()].reason_codes == ("evidence.p_value.unavailable",)
    assert rank.components["sampling"][0].context == {
        "reason": "rank_deficient",
        "metric": "unadjustable",
    }
    for metric, reason in (("rare", "weak_denominator"), ("zero", "zero_denominator")):
        row, cell = rows[metric], stored[metric]
        assert not row["sampling_available"] and row["point_available"] == (metric == "rare")
        context = {"metric": metric, "reason": reason, "arm": "B"}
        assert json.loads(row["failure_context"]) == context
        status, _ = cell.components["sampling"]
        assert status.state == "unavailable" and status.context == context
        verdict = verdicts[cell.identity.hash()]
        assert (verdict.status, verdict.reason_codes) == ("not_evaluated", (status.code,))
        key = cell.identity.key
        assert (key.aggregation, key.estimand) == ("ratio", "retained_window_ratio_difference")
        identity = cell.identity
        assert identity.metric_hash and identity.estimator_id and identity.method_config_hash
        assert identity.method_instance_id and identity.scope
        (history,) = store.history(
            "retained",
            HistorySelector(
                identity.metric_hash,
                identity.estimator_id,
                identity.method_config_hash,
                identity.method_instance_id,
                identity.scope.hash(),
            ),
        ).to_pylist()
        for view in (projected[metric], history):
            assert view["point"] == row["estimate"]
            assert view["reason_code"] == row["failure_code"]
            assert all(view[name] is None for name in ("lower", "upper", "se", "df"))
        assert row["p_value"] is None
        assert reason in report


@pytest.mark.integration
def test_all_rank_deficient_cuped_readout_roundtrips_without_an_estimate(tmp_path):
    experiment, reader = _experiment(
        [
            MetricSpec(
                name="adjusted",
                type="mean",
                value_column="y",
                decision_method=Method.cuped(covariates="constant"),
            ),
        ]
    )
    store = ParquetResultsStore(tmp_path)
    with reader:
        saved = run(experiment, reader, results_store=store)
    envelope = store.open_run("retained", saved.run.analysis_spec_hash, saved.run.run_id).envelope()
    (cell,) = envelope.cells
    assert cell.summary is not None and cell.summary.point is None
    assert cell.summary.reason == "rank_deficient"
    restored = ReadoutResults.model_validate_json(saved.readout().model_dump_json())
    (row,) = estimates_to_readout(restored)
    assert row["method"] == "switchback_cuped_jackknife_t"
    assert row["lift"] is row["standard_error"] is row["dof"] is row["stat_sig"] is None
    context = row["failure_context"]
    assert isinstance(context, str) and json.loads(context)["reason"] == "rank_deficient"
    assert row["available_cell_count"] == 0 and row["required_cell_count"] == 1


def test_changing_frozen_forecast_changes_method_identity_even_for_equal_values():
    identities = []
    for column in ("prior_position", "same_position"):
        experiment, reader = _experiment(
            [
                MetricSpec(
                    name="adjusted",
                    type="mean",
                    value_column="y",
                    decision_method=Method.cuped(covariates=column),
                ),
            ]
        )
        with reader:
            (cell,) = run(experiment, reader).cells
        assert cell.summary is not None and cell.summary.point == pytest.approx(7)
        identities.append(cell.identity.method_config_hash)
    assert all(identities) and identities[0] != identities[1]


@pytest.mark.parametrize(
    ("numerators", "point", "variance", "alpha"),
    [
        pytest.param((0, 2, 2, 4), 2, 2 / 3, 0.1, id="null-difference-overflows"),
        pytest.param((0, 0, 2, 2), 1, 1 / 3, 0.0025, id="half-width-overflows"),
    ],
)
def test_directional_ratio_tail_recovers_a_representable_standardized_margin(
    numerators, point, variance, alpha
):
    from fractions import Fraction
    from math import sqrt

    from scipy.stats import t

    from increment.decision import ContrastDecisionProcedure
    from increment.estimation.contrast import SwitchbackMetricStats, estimate_contrast
    from increment.estimation.decision_types import PValueEvidence
    from increment.estimation.results import contrast_stat_sig

    # Express the independent reference in units of 2**1022; both endpoints
    # of the comparison are finite, but the intermediate arithmetic need not be.
    unit = 2**1022
    stats = SwitchbackMetricStats(
        metric="rate",
        aggregation="ratio",
        probability_ct=0.5,
        randomization_law="independent_bernoulli_order",
        independence_grain="unit_cycle",
        n_units=4,
        n_cycles=4,
        control_group="A",
        treatment_group="B",
        observation_steps=1,
        carryover_order=0,
        retained_steps=1,
        ratio_components=tuple(
            (Fraction(v * unit), Fraction(1), Fraction(0), Fraction(1)) for v in numerators
        ),
    )
    procedure = ContrastDecisionProcedure(
        metric="rate",
        role="primary",
        alpha=alpha,
        alternative="greater",
        null_abs=(point - 4.5) * unit,
    )
    computed = estimate_contrast(stats, procedure)
    (row,) = computed.results
    (evidence,) = computed.evidence.values()
    assert isinstance(evidence, PValueEvidence)
    assert evidence.p_value == pytest.approx(t.sf(4.5 / sqrt(variance), 3))
    assert contrast_stat_sig(row) is True
    assert row.estimate is not None and row.estimate.lb is not None and row.estimate.ub is None
    assert row.estimate.lb / unit == pytest.approx(point - t.isf(alpha, 3) * sqrt(variance))


def test_zero_denominator_estimator_returns_only_the_keyed_failure():
    from increment.decision import ContrastDecisionProcedure
    from increment.declarations import ContrastHypothesisKey
    from increment.estimation.contrast import SwitchbackMetricStats, estimate_contrast

    stats = SwitchbackMetricStats.model_validate(
        {
            "metric": "sales",
            "aggregation": "ratio",
            "probability_ct": 0.5,
            "randomization_law": "independent_bernoulli_order",
            "independence_grain": "unit_cycle",
            "n_units": 2,
            "n_cycles": 2,
            "control_group": "old",
            "treatment_group": "new",
            "observation_steps": 1,
            "carryover_order": 0,
            "retained_steps": 1,
            "ratio_components": [(1, 1, 3, 0), (3, 1, 3, 0)],
        }
    )
    computed = estimate_contrast(
        stats,
        ContrastDecisionProcedure(
            metric="sales",
            role="primary",
            alpha=0.05,
            alternative="two-sided",
            null_abs=0,
        ),
    )
    assert not computed.results and not computed.evidence
    (key,) = computed.failures
    assert key == ContrastHypothesisKey("sales", "old", "new")
    assert computed.failures[key].context == {
        "metric": "sales",
        "reason": "zero_denominator",
        "arm": "old",
    }


@pytest.mark.parametrize("context", [None, {"metric": "sales", "reason": "unobserved_response"}])
def test_absent_measurements_require_the_recorded_missing_step(context):
    from increment.errors import InvalidRequestError
    from increment.results import ContrastResult

    with pytest.raises(InvalidRequestError) as caught:
        ContrastResult.model_validate(
            {
                "metric": "sales",
                "aggregation": "ratio",
                "probability_ct": 0.5,
                "method": "switchback_ratio_delta_t",
                "reference": "switchback_ratio_delta_t",
                "estimand": "retained_window_ratio_difference",
                "role": "primary",
                "alternative": "two-sided",
                "alpha": 0.05,
                "null_abs": 0,
                "randomization_law": "independent_bernoulli_order",
                "independence_grain": "unit_cycle",
                "n_units": 2,
                "n_cycles": 2,
                "control_group": "old",
                "treatment_group": "new",
                "observation_steps": 1,
                "carryover_order": 0,
                "retained_steps": 1,
                "estimate": None,
                "standard_error": None,
                "dof": None,
                "failure_code": "source.frame.switchback.response_unobserved" if context else None,
                "failure_context": context,
            }
        )
    assert caught.value.code == "estimation.contrast.reference_metadata"
