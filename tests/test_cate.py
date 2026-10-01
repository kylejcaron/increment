"""Tests for the public CATE adapters (``increment/cate.py``).

The adapters resolve source-backed inputs and validate source-only contracts
before passing arrays to the pure-math layer. The tests also assert that the
``unit_id`` column reaches validation and targeting as the honest split key.
"""

from __future__ import annotations

import copy
from typing import Literal, cast

import numpy as np
import pyarrow as pa
import pytest
from scipy.stats import norm

from increment import readouts, select_targeting_rule
from increment.cate import estimate_cate, targeting_rule, validate_cate
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.cate import CateResult, Covariate, fit_cate
from increment.estimation.targeting import CateValidation, TargetingRule
from increment.frame import MetricSpec, from_unit_summary, synthesise_metric
from increment.results import TargetingSelection
from increment.semantics.design import Randomized
from increment.sources import MomentsSource
from tests.test_sources import _plain_format_row

_N = 600
# The heterogeneous slope: one unit of spend above the sample mean buys this
# much extra effect; centering makes the sample-average effect exactly _TRUE_ATE.
_SLOPE = 0.5
_TRUE_ATE = 1.0


def _assert_cate_payload_close(expected, actual):
    """Compare portable public results, including nulls and exact provenance."""
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            _assert_cate_payload_close(value, actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_cate_payload_close(left, right)
    elif isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)
    else:
        assert actual == expected


def _assert_policy_roundtrip(policy, cols, ids):
    actions = policy.predict(cols, cluster_ids=ids)
    assert actions.shape == (len(next(iter(cols.values()))),)
    if policy.deploy_grain == "cluster":
        assert ids is not None
        for label in np.unique(ids):
            assert np.unique(actions[ids == label]).size == 1
    restored = TargetingRule.model_validate_json(policy.model_dump_json())
    np.testing.assert_array_equal(restored.predict(cols, cluster_ids=ids), actions)


def _source_parity_reports(source, frame, *, cluster_weight, clustered):
    from increment import ClusterBootstrap

    options: dict = {
        "control": "control",
        "interact": [],
        "cluster_weight": cluster_weight,
        "bootstrap": ClusterBootstrap(seed=57, repetitions=99),
    }
    validation = validate_cate(source, "revenue_per_user", n_groups=2, **options)
    rule = targeting_rule(source, "revenue_per_user", fraction=0.5, **options)
    selection = select_targeting_rule(
        source,
        "revenue_per_user",
        fractions=(0.0, 0.5, 1.0),
        n_folds=2,
        seed=12,
        **options,
    )
    assert validation.holdout_ate is not None
    assert validation.holdout_ate.lb is not None
    assert validation.holdout_ate.ub is not None
    assert validation.holdout_ate.lb < validation.holdout_ate.ub
    assert validation.n_clusters == rule.n_clusters
    if clustered:
        assert validation.holdout_ate_se is not None and validation.holdout_ate_se > 0
        assert validation.n_clusters is not None
        assert validation.n_clusters < validation.n_holdout
        assert validation.autoc.p_value is None
        assert validation.autoc.unavailable_reason == (
            "estimation.targeting.degenerate_rank_distribution"
        )
    else:
        assert validation.n_clusters is selection.n_clusters is None
    for policy in (rule, selection.rule):
        assert (
            policy.deploy_grain == policy.intervention_grain == ("cluster" if clustered else "unit")
        )
        _assert_policy_roundtrip(
            policy,
            {"spend": np.asarray(frame["spend"])},
            np.asarray(frame["store_id"]) if clustered else None,
        )
    return validation, rule, selection


def _frame(
    *,
    arms: tuple[str, ...] = ("control", "treatment"),
    spend_nulls: bool = False,
) -> pa.Table:
    """One row per user: variant, revenue, and two pre-exposure covariates."""
    rng = np.random.default_rng(20240517)
    variant = [arms[i % len(arms)] for i in range(_N)]
    spend = rng.normal(10.0, 3.0, _N)
    platform = np.array(["ios", "android", "web"])[rng.integers(0, 3, _N)]
    d = np.array([0.0 if v == arms[0] else 1.0 for v in variant])
    tau = _TRUE_ATE + _SLOPE * (spend - spend.mean())
    revenue = (
        20.0
        + 0.8 * spend
        + np.where(platform == "web", 1.5, 0.0)
        + tau * d
        + rng.normal(0.0, 2.0, _N)
    )
    spend_col = pa.array([None if spend_nulls and i == 7 else v for i, v in enumerate(spend)])
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(_N)],
            "variant": variant,
            "revenue": revenue,
            "spend": spend_col,
            "platform": platform.tolist(),
        }
    )


def _source(*, design=None, **kwargs):
    return from_unit_summary(
        _frame(**kwargs),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        design=design,
    )


def _source_from_frame(frame, *, design):
    return from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        design=design,
    )


# Ratio-metric source used to verify the CATE refusal and arm-level readout.
_BASE_RATIO = 4.0


def _frame_ratio(*, arms: tuple[str, ...] = ("control", "treatment")) -> pa.Table:
    """One row per user: variant, revenue (numerator), orders (denominator),
    and two pre-exposure covariates."""
    rng = np.random.default_rng(20240517)
    variant = [arms[i % len(arms)] for i in range(_N)]
    spend = rng.normal(10.0, 3.0, _N)
    platform = np.array(["ios", "android", "web"])[rng.integers(0, 3, _N)]
    d = np.array([0.0 if v == arms[0] else 1.0 for v in variant])
    orders = rng.gamma(shape=9.0, scale=1.0, size=_N)
    r = _BASE_RATIO + np.where(platform == "web", 0.4, 0.0) + d
    revenue = r * orders + rng.normal(0.0, 1.0, _N)
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(_N)],
            "variant": variant,
            "revenue": revenue,
            "orders": orders,
            "spend": spend,
            "platform": platform.tolist(),
        }
    )


def _source_ratio(**kwargs):
    return from_unit_summary(
        _frame_ratio(**kwargs),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="aov", type="ratio", numerator="revenue", denominator="orders")],
        design=Randomized(control_group="control"),
    )


class TestEstimate:
    def test_recovers_the_known_ate(self):
        result = estimate_cate(_source(), "revenue", control="control", interact=["spend"])
        assert isinstance(result, CateResult)
        assert abs(result.ate - _TRUE_ATE) < 3.0 * result.se
        assert result.n == _N
        assert result.n_treated + result.n_control == _N

    def test_detects_the_planted_heterogeneity(self):
        result = estimate_cate(_source(), "revenue", control="control", interact=["spend"])
        assert result.heterogeneity.df == 1
        assert result.heterogeneity.p_value < 0.01
        (effect,) = result.interactions
        assert effect.name == "d:spend"
        # The interaction is on the standardized basis: a one-sd move in
        # spend buys _SLOPE * sd(spend) of extra effect.
        expected = _SLOPE * float(np.asarray(_frame()["spend"]).std(ddof=1))
        assert effect.lb < expected < effect.ub

    def test_string_and_covariate_spellings_agree(self):
        src = _source()
        as_string = estimate_cate(src, "revenue", control="control", interact=["spend"])
        as_typed = estimate_cate(
            src, "revenue", control="control", interact=[Covariate(name="spend")]
        )
        assert as_typed == as_string

    def test_categorical_adjustment_enters_as_a_main_effect(self):
        """An adjust-only covariate widens the main block, never the interactions."""
        result = estimate_cate(
            _source(),
            "revenue",
            control="control",
            interact=["spend"],
            adjust=[Covariate(name="platform", kind="categorical")],
        )
        assert [e.name for e in result.interactions] == ["d:spend"]
        # Three levels, one absorbed as the modal reference -> two one-hot columns.
        assert sum(c.startswith("platform=") for c in result.columns) == 2
        assert abs(result.ate - _TRUE_ATE) < 3.0 * result.se

    def test_ard_shrinks_the_score_without_moving_the_ate_or_the_test(self):
        """``ard=True`` reaches ``fit_cate``'s shrinkage through the adapter.

        The interaction block is deliberately wider than the truth - only
        ``spend`` moves the effect, so the ``platform`` columns are noise
        for ARD to pull toward zero, without moving the ATE, its SE, or
        the joint heterogeneity test.
        """
        src = _source()
        interact = ["spend", Covariate(name="platform", kind="categorical")]
        plain = estimate_cate(src, "revenue", control="control", interact=interact)
        shrunk = estimate_cate(src, "revenue", control="control", interact=interact, ard=True)

        assert shrunk.ate == plain.ate
        assert shrunk.se == plain.se
        assert shrunk.heterogeneity == plain.heterogeneity
        assert [e.coef for e in shrunk.interactions] == [e.coef for e in plain.interactions]
        assert all(e.ard_coef is not None for e in shrunk.interactions)

        frame = _frame()
        cols = {"spend": np.asarray(frame["spend"]), "platform": np.asarray(frame["platform"])}
        assert not np.allclose(
            shrunk.score(cols, deploy_grain="unit"), plain.score(cols, deploy_grain="unit")
        )
        # Shrinkage, not merely a different number: the scored spread contracts.
        assert (
            shrunk.score(cols, deploy_grain="unit").std()
            < plain.score(cols, deploy_grain="unit").std()
        )


def test_cate_inference_is_invariant_to_representable_outcome_translation():
    n = 120
    d = (np.arange(n) % 2).astype(float)
    base = (np.arange(n) % 7 + 2 * d).astype(float)
    x = (np.arange(n) % 5).astype(float)
    cols = {"x": x}
    clusters = np.arange(n) // 4

    plain = fit_cate(base, d, {}, interact=())
    assert plain.ate == pytest.approx(1.95)
    assert plain.se == pytest.approx(0.3700854993594832)
    assert plain.cate({}).value == pytest.approx(plain.ate)

    interactions = (Covariate(name="x"),)
    cases = (
        ((), False, None),
        (interactions, False, None),
        (interactions, True, None),
        (interactions, False, clusters),
        (interactions, True, clusters),
    )
    for interact, ard, cluster_ids in cases:
        inputs = cols if interact else {}
        reference = fit_cate(base, d, inputs, interact=interact, ard=ard, cluster_ids=cluster_ids)
        for shift in (1e12, 1e15):
            translated = fit_cate(
                base + shift,
                d,
                inputs,
                interact=interact,
                ard=ard,
                cluster_ids=cluster_ids,
            )
            np.testing.assert_allclose(
                [translated.ate, translated.se, translated.lb, translated.ub],
                [reference.ate, reference.se, reference.lb, reference.ub],
                rtol=1e-12,
                atol=1e-12,
            )
            point = {"x": 2.0} if interact else {}
            np.testing.assert_allclose(
                translated.cate(point).value,
                reference.cate(point).value,
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                translated.score(inputs, cluster_ids=cluster_ids, deploy_grain="unit"),
                reference.score(inputs, cluster_ids=cluster_ids, deploy_grain="unit"),
                rtol=1e-12,
                atol=1e-12,
            )


class TestRefusals:
    def test_moments_only_source_cannot_serve_unit_grain(self):
        src = MomentsSource(
            [_plain_format_row(metric="revenue", group_id="control", n=10, ref_y=0.1)],
            metrics=[synthesise_metric(MetricSpec(name="revenue"))],
            study_id="cube",
            design=Randomized(control_group="control"),
        )
        with pytest.raises(CapabilityError) as raised:
            estimate_cate(src, "revenue", control="control", interact=["spend"])
        assert raised.value.code == "source.moments.covariate_unavailable"

    def test_unknown_metric_is_named(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_cate(_source(), "orders", control="control", interact=["spend"])
        assert exc_info.value.code == "cate.metric_declared_source"
        assert exc_info.value.context["metric"] == "orders"

    def test_null_covariate_is_refused_by_name(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_cate(
                _source(spend_nulls=True), "revenue", control="control", interact=["spend"]
            )
        assert exc_info.value.code == "estimation.cate.covariate_nulls_non"
        assert exc_info.value.context["name"] == "spend"

    def test_three_arms_refuse_rather_than_pool(self):
        from increment.errors import InvalidRequestError

        src = _source(arms=("control", "treatment", "treatment_b"))
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_cate(src, "revenue", control="control", interact=["spend"])
        assert exc_info.value.code == "cate.contrasts_exactly_two"
        assert exc_info.value.context["others"] == ("treatment", "treatment_b")

    def test_absent_control_is_refused(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_cate(_source(), "revenue", control="holdout", interact=["spend"])
        assert exc_info.value.code == "cate.control_arm_present"
        assert exc_info.value.context["control"] == "holdout"

    def test_quantile_metric_refuses_for_all_four_entry_points(self):
        """unit_frame's y is the unit TOTAL, so a fit would be a
        conditional MEAN effect reported under a quantile metric's name -
        the only remaining metric-type refusal at this adapter seam."""
        src = from_unit_summary(
            _frame(),
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", type="quantile", quantile=0.5)],
        )
        from increment.errors import UnsupportedRequestError

        with pytest.raises(UnsupportedRequestError) as exc_info:
            estimate_cate(src, "revenue", control="control", interact=["spend"])
        assert exc_info.value.code == "cate.cate_supported_quantile"
        with pytest.raises(UnsupportedRequestError) as exc_info:
            validate_cate(src, "revenue", control="control", interact=["spend"])
        assert exc_info.value.code == "cate.cate_supported_quantile"
        with pytest.raises(UnsupportedRequestError) as exc_info:
            targeting_rule(src, "revenue", control="control", interact=["spend"], fraction=0.2)
        assert exc_info.value.code == "cate.cate_supported_quantile"
        with pytest.raises(UnsupportedRequestError) as exc_info:
            select_targeting_rule(
                src,
                "revenue",
                control="control",
                interact=["spend"],
                fractions=(0.2,),
                seed=11,
            )
        assert exc_info.value.code == "cate.cate_supported_quantile"

    @pytest.mark.parametrize(
        ("caller", "entry_point", "kwargs"),
        [
            (
                "estimate_cate",
                estimate_cate,
                {"control": "control", "interact": ["spend"]},
            ),
            (
                "validate_cate",
                validate_cate,
                {"control": "control", "interact": ["spend"]},
            ),
            (
                "targeting_rule",
                targeting_rule,
                {"control": "control", "interact": ["spend"], "fraction": 0.2},
            ),
            (
                "select_targeting_rule",
                select_targeting_rule,
                {
                    "control": "control",
                    "interact": ["spend"],
                    "fractions": (0.2,),
                    "seed": 11,
                },
            ),
        ],
    )
    def test_ratio_metric_refuses_before_unit_frame(self, caller, entry_point, kwargs, monkeypatch):
        from increment.errors import UnsupportedRequestError

        src = _source_ratio()

        def fail_if_called(*args, **kwargs):
            raise AssertionError("ratio refusal must precede unit_frame")

        monkeypatch.setattr(src, "unit_frame", fail_if_called)

        with pytest.raises(UnsupportedRequestError) as exc_info:
            entry_point(src, "aov", **kwargs)
        assert exc_info.value.code == "cate.does_support_ratio"
        assert exc_info.value.context["caller"] == caller
        assert exc_info.value.context["metric"] == "aov"

    def test_arm_level_ratio_readout_remains_supported(self):
        (row,) = [row for row in readouts.run(_source_ratio()) if row.metric == "aov"]
        assert row.abs_diff is not None


def test_estimate_cate_refuses_non_randomized_source():
    from increment.errors import InvalidRequestError
    from increment.semantics.design import AdjustmentSet, Observational

    src = _source(
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("spend",))
        )
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_cate(src, "revenue", control="control", interact=["spend"])
    assert exc_info.value.code == "cate.identification.randomized_only"


@pytest.mark.parametrize("caller", ["validate_cate", "targeting_rule", "select_targeting_rule"])
def test_targeting_entry_points_accept_numeric_adjustment_only_with_categorical_cate(caller):
    from increment.semantics.design import AdjustmentSet, Observational

    src = _source(
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("spend",))
        )
    )
    result = _call_targeting_entry_point(
        caller,
        src,
        interact=[Covariate(name="platform", kind="categorical")],
    )

    assert isinstance(result, CateValidation | TargetingRule | TargetingSelection)
    population = (
        result.rule.population if isinstance(result, TargetingSelection) else result.population
    )
    assert population is None


def _call_targeting_entry_point(caller, src, *, interact=None):
    interact = interact or ["spend"]
    if caller == "validate_cate":
        return validate_cate(src, "revenue", control="control", interact=interact)
    if caller == "targeting_rule":
        return targeting_rule(src, "revenue", control="control", interact=interact, fraction=0.2)
    return select_targeting_rule(
        src,
        "revenue",
        control="control",
        interact=interact,
        fractions=(0.2, 0.4),
        seed=1,
    )


@pytest.mark.parametrize("caller", ["validate_cate", "targeting_rule", "select_targeting_rule"])
@pytest.mark.parametrize("source_kind", ["arrow_decimal", "pandas_nullable"])
def test_targeting_facade_preserves_numeric_adjustment_dtype(caller, source_kind, monkeypatch):
    from decimal import Decimal

    from increment.semantics.design import AdjustmentSet, Observational

    frame = _frame()
    if source_kind == "arrow_decimal":
        spend = pa.array(
            [Decimal(i % 17) / Decimal(10) for i in range(_N)],
            type=pa.decimal128(8, 2),
        )
        frame = frame.set_column(frame.schema.get_field_index("spend"), "spend", spend)
    else:
        import pandas as pd

        frame = frame.to_pandas()
        frame["spend"] = pd.array(np.arange(_N) % 17, dtype="Int64")
    src = _source_from_frame(
        frame,
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("spend",))
        ),
    )
    seen: list[np.ndarray] = []
    sentinel = object()

    def capture(_y, _d, cols, _unit_ids, **_kwargs):
        seen.append(cols["spend"])
        return sentinel

    downstream = {
        "validate_cate": "validate_cate_arrays",
        "targeting_rule": "targeting_rule_arrays",
        "select_targeting_rule": "select_targeting_rule_arrays",
    }[caller]
    monkeypatch.setattr(f"increment.cate.{downstream}", capture)

    assert _call_targeting_entry_point(caller, src) is sentinel
    assert len(seen) == 1
    assert seen[0].dtype == np.float64


@pytest.mark.parametrize("source_kind", ["arrow_decimal", "pandas_nullable"])
def test_targeting_facade_preserves_numeric_adjustment_null_refusal(source_kind):
    from decimal import Decimal

    from increment.errors import InvalidRequestError
    from increment.semantics.design import AdjustmentSet, Observational

    frame = _frame()
    if source_kind == "arrow_decimal":
        values = [None if i == 7 else Decimal(i % 17) / Decimal(10) for i in range(_N)]
        spend = pa.array(values, type=pa.decimal128(8, 2))
        frame = frame.set_column(frame.schema.get_field_index("spend"), "spend", spend)
    else:
        import pandas as pd

        frame = frame.to_pandas()
        values = np.arange(_N) % 17
        frame["spend"] = pd.array(values, dtype="Int64")
        frame.loc[7, "spend"] = pd.NA
    src = _source_from_frame(
        frame,
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("spend",))
        ),
    )

    with pytest.raises(InvalidRequestError) as exc_info:
        validate_cate(src, "revenue", control="control", interact=["spend"])
    assert exc_info.value.code == "estimation.cate.covariate_nulls_non"
    assert exc_info.value.context["name"] == "spend"


@pytest.mark.parametrize("caller", ["validate_cate", "targeting_rule", "select_targeting_rule"])
def test_targeting_categorical_observational_adjustment_matches_dummies(caller):
    """A raw ``platform`` string in the declared adjustment set reproduces the
    hand-built modal-reference dummies row for row, and the same column may
    still serve as a categorical CATE modifier."""
    from increment.semantics.design import AdjustmentSet, Observational
    from tests.categorical_cases import assert_rows_match

    frame = _frame()
    platform = np.asarray(frame["platform"].to_pylist())
    levels, counts = np.unique(platform, return_counts=True)
    order = sorted(range(len(levels)), key=lambda i: (-counts[i], levels[i]))
    dummies = frame
    dummy_names = []
    for i in order[1:]:
        name = f"platform_{levels[i]}"
        dummy_names.append(name)
        dummies = dummies.append_column(name, pa.array((platform == levels[i]).astype(float)))
    raw_src = _source_from_frame(
        frame,
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("spend", "platform"))
        ),
    )
    oracle_src = _source_from_frame(
        dummies,
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("spend", *dummy_names)),
        ),
    )
    interact = [Covariate(name="platform", kind="categorical"), "spend"]
    oracle = _call_targeting_entry_point(caller, oracle_src, interact=interact).model_dump()
    actual = _call_targeting_entry_point(caller, raw_src, interact=interact).model_dump()
    # Fold-specific modal references span the same space; standardized ridge
    # regularization (1e-6) makes their predictions only approximately invariant.
    assert_rows_match(oracle, actual, rel=1e-6, skip=("required_columns",))


@pytest.mark.parametrize("caller", ["validate_cate", "targeting_rule", "select_targeting_rule"])
def test_targeting_categorical_observational_adjustment_null_refuses_before_fit(
    caller, monkeypatch
):
    """This path supports missing='refuse' only: a null level is a missing
    covariate value, refused by name before any model is fitted."""
    from increment.errors import InvalidRequestError
    from increment.semantics.design import AdjustmentSet, Observational

    frame = _frame()
    platform = frame["platform"].to_pylist()
    platform[7] = None
    frame = frame.set_column(
        frame.schema.get_field_index("platform"), "platform", pa.array(platform)
    )
    src = _source_from_frame(
        frame,
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("platform",))
        ),
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("CATE model fitted before validating observational adjustment")

    monkeypatch.setattr("increment.estimation.targeting.fit_cate", fail)
    with pytest.raises(InvalidRequestError) as exc_info:
        _call_targeting_entry_point(caller, src)
    assert exc_info.value.code == "estimation.cate.covariate_nulls"
    assert exc_info.value.context["name"] == "platform"


@pytest.mark.parametrize("caller", ["validate_cate", "targeting_rule", "select_targeting_rule"])
@pytest.mark.parametrize("missing", ["impute-indicator", "pattern", "complete-case", "allow"])
def test_targeting_refuses_unsupported_missing_policy_before_unit_frame(
    caller, missing, monkeypatch
):
    from increment.errors import InvalidRequestError
    from increment.semantics.design import AdjustmentSet, Observational

    src = _source(
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("spend",), missing=missing),
        )
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("unit_frame accessed before the missing-policy refusal")

    monkeypatch.setattr(src, "unit_frame", fail)
    with pytest.raises(InvalidRequestError) as exc_info:
        _call_targeting_entry_point(caller, src)
    assert exc_info.value.code == "cate.identification.unsupported_missing_policy"
    assert exc_info.value.context["caller"] == caller
    assert exc_info.value.context["missing"] == missing


@pytest.mark.parametrize("caller", ["validate_cate", "targeting_rule", "select_targeting_rule"])
def test_targeting_refuses_unsupported_max_smd_before_unit_frame(caller, monkeypatch):
    from increment.errors import InvalidRequestError
    from increment.semantics.design import AdjustmentSet, IdentificationGate, Observational

    src = _source(
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("spend",)),
            gate=IdentificationGate(max_smd=0.1),
        )
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("unit_frame accessed before the balance-policy refusal")

    monkeypatch.setattr(src, "unit_frame", fail)
    with pytest.raises(InvalidRequestError) as exc_info:
        _call_targeting_entry_point(caller, src)
    assert exc_info.value.code == "cate.identification.unsupported_max_smd"
    assert exc_info.value.context["caller"] == caller
    assert exc_info.value.context["max_smd"] == 0.1


def test_estimate_refuses_non_randomized_before_touching_unit_frame(monkeypatch):
    from increment.errors import InvalidRequestError
    from increment.semantics.design import AdjustmentSet, Observational

    src = _source(
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("spend",))
        )
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("unit_frame accessed before the mechanism check")

    monkeypatch.setattr(src, "unit_frame", fail)
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_cate(src, "revenue", control="control", interact=["spend"])
    assert exc_info.value.code == "cate.identification.randomized_only"


def test_unit_design_refuses_a_design_less_source_before_touching_unit_frame(monkeypatch):
    """No design means no identification claim; refuse before any row is read."""
    from increment.errors import InvalidRequestError

    src = MomentsSource(
        [_plain_format_row(metric="revenue", group_id="control", n=10, ref_y=0.1)],
        metrics=[synthesise_metric(MetricSpec(name="revenue"))],
        study_id="cube",
    )
    assert src.context.design is None

    def fail(*_args, **_kwargs):
        raise AssertionError("unit_frame accessed before the mechanism check")

    monkeypatch.setattr(src, "unit_frame", fail)
    with pytest.raises(InvalidRequestError) as exc_info:
        validate_cate(src, "revenue", control="control", interact=["spend"])
    assert exc_info.value.code == "cate.identification.unsupported_mechanism"
    assert exc_info.value.context["mechanism"] is None


class TestValidate:
    def test_reports_an_honest_split_of_every_unit(self):
        result = validate_cate(_source(), "revenue", control="control", interact=["spend"])
        assert isinstance(result, CateValidation)
        assert result.n_train + result.n_holdout == _N
        # A crc32 split of these ids happens to come out even; the guarantee
        # is only that neither half collapses.
        assert min(result.n_train, result.n_holdout) > _N // 3
        assert len(result.groups) == 5
        assert sum(g.n for g in result.groups) == result.n_holdout
        # holdout_ate is an UNADJUSTED difference in means on half the units,
        # so it still carries the prognostic spend imbalance the fit removes.
        ate = result.holdout_ate
        assert ate is not None
        assert ate.lb is not None and ate.ub is not None
        ate_se = (ate.ub - ate.lb) / (2.0 * norm.ppf(0.975))
        assert abs(ate.value - _TRUE_ATE) < 3.0 * ate_se

    def test_detects_the_planted_heterogeneity_out_of_sample(self):
        result = validate_cate(_source(), "revenue", control="control", interact=["spend"])
        assert result.autoc.estimate is not None
        assert result.autoc.estimate > 0.0
        assert result.autoc.p_value is not None
        assert result.autoc.p_value < 0.05
        assert result.passed
        # Spend drives the effect, so the most-affected group spends more.
        (row,) = [r for r in result.clan if r.covariate == "spend"]
        assert row.lb is not None
        assert row.lb > 0.0

    def test_n_groups_and_alpha_reach_the_estimation_layer(self):
        result = validate_cate(
            _source(), "revenue", control="control", interact=["spend"], n_groups=3, alpha=0.01
        )
        assert [g.group for g in result.groups] == [1, 2, 3]
        assert result.alpha == 0.01
        assert result.autoc.p_value is not None
        assert result.passed is (result.autoc.p_value < 0.01)

    def test_string_and_covariate_spellings_agree(self):
        src = _source()
        as_string = validate_cate(src, "revenue", control="control", interact=["spend"])
        as_typed = validate_cate(
            src, "revenue", control="control", interact=[Covariate(name="spend")]
        )
        assert as_typed == as_string


class TestValidateRefusals:
    def test_moments_only_source_cannot_serve_unit_grain(self):
        src = MomentsSource(
            [_plain_format_row(metric="revenue", group_id="control", n=10, ref_y=0.1)],
            metrics=[synthesise_metric(MetricSpec(name="revenue"))],
            study_id="cube",
            design=Randomized(control_group="control"),
        )
        with pytest.raises(CapabilityError) as raised:
            validate_cate(src, "revenue", control="control", interact=["spend"])
        assert raised.value.code == "source.moments.covariate_unavailable"

    def test_unknown_metric_is_named(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate(_source(), "orders", control="control", interact=["spend"])
        assert exc_info.value.code == "cate.metric_declared_source"
        assert exc_info.value.context["metric"] == "orders"

    def test_three_arms_refuse_rather_than_pool(self):
        from increment.errors import InvalidRequestError

        src = _source(arms=("control", "treatment", "treatment_b"))
        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate(src, "revenue", control="control", interact=["spend"])
        assert exc_info.value.code == "cate.contrasts_exactly_two"
        assert exc_info.value.context["caller"] == "validate_cate"

    def test_absent_control_is_refused(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate(_source(), "revenue", control="holdout", interact=["spend"])
        assert exc_info.value.code == "cate.control_arm_present"
        assert exc_info.value.context["control"] == "holdout"


class TestTargetingRule:
    def test_the_gate_passes_and_the_rule_is_deployable(self):
        rule = targeting_rule(
            _source(), "revenue", control="control", interact=["spend"], fraction=0.25
        )
        assert isinstance(rule, TargetingRule)
        assert rule.recommendation == "target"
        assert rule.fraction == 0.25
        assert rule.threshold is not None
        assert rule.policy_value is not None
        assert rule.uplift_vs_average is not None
        # Spend drives the effect, so the quarter above the cut beats the average.
        assert rule.validation.holdout_ate is not None
        assert rule.policy_value.value > rule.validation.holdout_ate.value
        assert rule.uplift_vs_average.value > 0.0

    def test_the_attached_validation_is_what_validate_cate_reports(self):
        src = _source()
        rule = targeting_rule(src, "revenue", control="control", interact=["spend"], fraction=0.25)
        assert rule.validation == validate_cate(
            src, "revenue", control="control", interact=["spend"]
        )

    def test_a_gate_that_cannot_be_cleared_returns_the_simple_policy(self):
        """A refusal is a value, not an exception - with the evidence attached."""
        rule = targeting_rule(
            _source(),
            "revenue",
            control="control",
            interact=["spend"],
            fraction=0.25,
            alpha=1e-9,
        )
        assert rule.recommendation == "simple"
        assert rule.threshold is None
        assert rule.policy_value is None
        assert rule.uplift_vs_average is None
        assert isinstance(rule.validation, CateValidation)
        assert not rule.validation.passed
        assert rule.validation.n_train + rule.validation.n_holdout == _N

    def test_string_and_covariate_spellings_agree(self):
        src = _source()
        as_string = targeting_rule(
            src, "revenue", control="control", interact=["spend"], fraction=0.25
        )
        as_typed = targeting_rule(
            src,
            "revenue",
            control="control",
            interact=[Covariate(name="spend")],
            fraction=0.25,
        )
        assert as_typed == as_string

    def test_fraction_is_required(self):
        with pytest.raises(TypeError):
            targeting_rule(_source(), "revenue", control="control", interact=["spend"])  # ty: ignore


class TestTargetingRuleRefusals:
    def test_moments_only_source_cannot_serve_unit_grain(self):
        src = MomentsSource(
            [_plain_format_row(metric="revenue", group_id="control", n=10, ref_y=0.1)],
            metrics=[synthesise_metric(MetricSpec(name="revenue"))],
            study_id="cube",
            design=Randomized(control_group="control"),
        )
        with pytest.raises(CapabilityError) as raised:
            targeting_rule(src, "revenue", control="control", interact=["spend"], fraction=0.25)
        assert raised.value.code == "source.moments.covariate_unavailable"

    def test_unknown_metric_is_named(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            targeting_rule(_source(), "orders", control="control", interact=["spend"], fraction=0.2)
        assert exc_info.value.code == "cate.metric_declared_source"
        assert exc_info.value.context["metric"] == "orders"

    def test_three_arms_refuse_rather_than_pool(self):
        from increment.errors import InvalidRequestError

        src = _source(arms=("control", "treatment", "treatment_b"))
        with pytest.raises(InvalidRequestError) as exc_info:
            targeting_rule(src, "revenue", control="control", interact=["spend"], fraction=0.25)
        assert exc_info.value.code == "cate.contrasts_exactly_two"
        assert exc_info.value.context["caller"] == "targeting_rule"

    def test_absent_control_is_refused(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            targeting_rule(
                _source(), "revenue", control="holdout", interact=["spend"], fraction=0.2
            )
        assert exc_info.value.code == "cate.control_arm_present"
        assert exc_info.value.context["control"] == "holdout"

    def test_a_fraction_outside_the_unit_interval_is_refused(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            targeting_rule(
                _source(), "revenue", control="control", interact=["spend"], fraction=-0.1
            )
        assert exc_info.value.code == "estimation.targeting.fraction_share_units"


class TestSelectTargetingRule:
    def test_returns_a_selection_with_a_deployable_rule(self):
        result = select_targeting_rule(
            _source(),
            "revenue",
            control="control",
            interact=["spend"],
            fractions=(0.1, 0.25, 0.5),
            seed=11,
        )
        assert isinstance(result, TargetingSelection)
        assert result.selected_fraction in result.fractions
        assert result.rule.fraction == result.selected_fraction

    def test_seed_is_required(self):
        with pytest.raises(TypeError):
            select_targeting_rule(  # ty: ignore
                _source(), "revenue", control="control", interact=["spend"], fractions=(0.2,)
            )

    def test_fractions_are_required(self):
        with pytest.raises(TypeError):
            select_targeting_rule(  # ty: ignore
                _source(), "revenue", control="control", interact=["spend"], seed=11
            )

    def test_moments_only_sources_are_refused(self):
        src = MomentsSource(
            [_plain_format_row(metric="revenue", group_id="control", n=10, ref_y=0.1)],
            metrics=[synthesise_metric(MetricSpec(name="revenue"))],
            study_id="cube",
            design=Randomized(control_group="control"),
        )
        with pytest.raises(CapabilityError) as raised:
            select_targeting_rule(
                src,
                "revenue",
                control="control",
                interact=["spend"],
                fractions=(0.2,),
                seed=11,
            )
        assert raised.value.code == "source.moments.covariate_unavailable"


_N_CLUSTERS = 100
_PER_CLUSTER = 6  # -> 600 rows, matching _N


def _cluster_label(index: int, dtype: str) -> str:
    if dtype == "numeric_looking":
        return str(1000 + index)
    if dtype == "uuid":
        import uuid

        return str(uuid.UUID(int=index))
    return f"store_{index}"


def _cluster_frame(*, pure_arms: bool = True, cluster_dtype: str = "str") -> pa.Table:
    """Clustered variant of ``_frame``: ``_PER_CLUSTER`` units share one cluster id.

    ``pure_arms=True`` assigns every cluster fully to one arm (declared
    cluster-randomization); ``pure_arms=False`` alternates arm within a
    cluster (mixed-treatment observational dependence).
    """
    rng = np.random.default_rng(20240517)
    user_id, variant, revenue, spend, platform, cluster_id = [], [], [], [], [], []
    for c in range(_N_CLUSTERS):
        cluster_shock = rng.normal(0.0, 1.0)
        cluster_arm = "control" if c % 2 == 0 else "treatment"
        label = _cluster_label(c, cluster_dtype)
        for u in range(_PER_CLUSTER):
            arm = cluster_arm if pure_arms else ("control" if u % 2 == 0 else "treatment")
            s = rng.normal(10.0, 3.0)
            p = ["ios", "android", "web"][rng.integers(0, 3)]
            d = 0.0 if arm == "control" else 1.0
            tau = _TRUE_ATE + _SLOPE * (s - 10.0)
            y = 20.0 + 0.8 * s + cluster_shock + tau * d + rng.normal(0.0, 1.0)
            user_id.append(f"u{c}_{u}")
            variant.append(arm)
            revenue.append(y)
            spend.append(s)
            platform.append(p)
            cluster_id.append(label)
    return pa.table(
        {
            "user_id": user_id,
            "variant": variant,
            "revenue": revenue,
            "spend": spend,
            "platform": platform,
            "store_id": cluster_id,
        }
    )


def _cluster_source(
    *,
    pure_arms: bool = True,
    cluster_dtype: str = "str",
    design=None,
    intervention_grain: Literal["unit", "cluster"] = "unit",
    cluster: str | None = "store_id",
):
    return from_unit_summary(
        _cluster_frame(pure_arms=pure_arms, cluster_dtype=cluster_dtype),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        cluster=cluster,
        design=design,
        intervention_grain=intervention_grain,
    )


class TestClusterTransport:
    """Cluster provenance and cluster-atomic validation through public adapters."""

    def test_unclustered_fit_reports_no_cluster_count(self):
        result = estimate_cate(_source(), "revenue", control="control", interact=["spend"])
        assert result.model_dump()["n_clusters"] is None

    @pytest.mark.parametrize("dtype", ["str", "uuid", "numeric_looking"])
    def test_string_uuid_and_numeric_looking_cluster_ids_never_become_covariates(self, dtype):
        frame = _cluster_frame(cluster_dtype=dtype).to_pandas()
        if dtype == "uuid":
            from uuid import UUID

            frame["store_id"] = [UUID(value) for value in frame["store_id"]]
        clustered = from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue")],
            cluster="store_id",
        )
        independent = from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue")],
        )
        result = estimate_cate(clustered, "revenue", control="control", interact=["spend"])
        reference = estimate_cate(independent, "revenue", control="control", interact=["spend"])
        assert result.n_clusters == _N_CLUSTERS
        assert result.ate == pytest.approx(reference.ate)

    def test_row_permutation_preserves_cluster_alignment_and_results(self):
        frame = _cluster_frame()
        shuffled = frame.take(np.random.default_rng(7).permutation(frame.num_rows))
        ordered_src = from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue")],
            cluster="store_id",
        )
        shuffled_src = from_unit_summary(
            shuffled,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue")],
            cluster="store_id",
        )
        ordered = estimate_cate(ordered_src, "revenue", control="control", interact=["spend"])
        permuted = estimate_cate(shuffled_src, "revenue", control="control", interact=["spend"])
        assert permuted.ate == pytest.approx(ordered.ate, rel=1e-9)
        assert permuted.se == pytest.approx(ordered.se, rel=1e-9)

        ordered_validation = validate_cate(
            ordered_src, "revenue", control="control", interact=["spend"]
        )
        permuted_validation = validate_cate(
            shuffled_src, "revenue", control="control", interact=["spend"]
        )
        assert permuted_validation.n_holdout == ordered_validation.n_holdout
        assert permuted_validation.n_clusters == ordered_validation.n_clusters
        assert permuted_validation.autoc.estimate == pytest.approx(
            ordered_validation.autoc.estimate
        )
        assert permuted_validation.autoc.p_value == pytest.approx(ordered_validation.autoc.p_value)

    def test_missing_cluster_id_refuses_before_fit(self):
        """A null cluster label refuses at source construction -- before any
        estimator ever reads a row -- via the frame path's own data-quality
        gate (`source.frame.cluster_labels`); the crossfit-layer missing-id
        refusal is the second line of defense for a source that skips it."""
        frame = _cluster_frame()
        table = frame.to_pandas()
        table.loc[0, "store_id"] = None
        with pytest.raises(InvalidRequestError) as exc_info:
            from_unit_summary(
                table,
                unit="user_id",
                group="variant",
                control="control",
                metrics=[MetricSpec(name="revenue")],
                cluster="store_id",
            )
        assert exc_info.value.code == "source.frame.cluster_labels"

    def test_colliding_native_cluster_identities_refuse_before_fit(self):
        """Genuine ingress-level proof, no monkeypatching: a raw source
        column that mixes native int and str values for the same canonical
        string must refuse via real canonicalization, not a post-hoc
        string cast that would silently erase the distinction first."""
        table = _cluster_frame().to_pandas()
        table["store_id"] = table["store_id"].astype(object)
        # Two distinct native values -- str "1" and int 1 -- render the same
        # canonical string; both remain singleton, pure-arm clusters under
        # frame.py's own construction-time checks, so only cluster-identity
        # canonicalization can catch this.
        table.loc[0, "store_id"] = "1"
        table.loc[6, "store_id"] = 1
        src = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue")],
            cluster="store_id",
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_cate(src, "revenue", control="control", interact=["spend"])
        assert exc_info.value.code == "estimation.crossfit.identity_collision"

    def test_cluster_column_cannot_be_requested_as_interact_covariate(self):
        src = _cluster_source()
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_cate(src, "revenue", control="control", interact=["store_id"])
        assert exc_info.value.code == "cate.cluster.covariate_conflict"
        assert exc_info.value.context["caller"] == "estimate_cate"
        assert exc_info.value.context["column"] == "store_id"

    def test_cluster_column_cannot_be_requested_as_adjust_covariate(self):
        src = _cluster_source()
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_cate(
                src, "revenue", control="control", interact=["spend"], adjust=["store_id"]
            )
        assert exc_info.value.code == "cate.cluster.covariate_conflict"

    def test_declared_cluster_randomization_refuses_a_cluster_spanning_both_arms(self):
        """A declared cluster-randomization (default Randomized design) with
        an impure cluster refuses at source construction -- the frame path's
        own data-quality gate, which fires before any estimator call."""
        with pytest.raises(InvalidRequestError) as exc_info:
            _cluster_source(pure_arms=False)
        assert exc_info.value.code == "source.frame.cluster_labels"

    def test_unit_design_refuses_impure_clusters_under_a_randomized_mechanism(self, monkeypatch):
        """Belt-and-suspenders: `_unit_design`'s own arm-purity guard fires
        even when a source's construction-time check was bypassed (built
        observational, then queried as randomized) -- the estimator-level
        defense the frame path's construction-time gate front-runs for a
        well-behaved, consistently-declared source."""
        import dataclasses

        from increment.semantics.design import AdjustmentSet, Observational

        src = _cluster_source(
            pure_arms=False,
            design=Observational(
                control_group="control", adjustment=AdjustmentSet(covariates=("spend",))
            ),
        )
        randomized = Randomized(control_group="control")
        monkeypatch.setattr(src, "_context", dataclasses.replace(src.context, design=randomized))
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_cate(src, "revenue", control="control", interact=["spend"])
        assert exc_info.value.code == "cate.cluster.randomization_arm_purity"

    def test_clustered_fit_reports_cluster_count_and_deployment_grain(self):
        src = _cluster_source(pure_arms=True, intervention_grain="cluster")
        result = estimate_cate(src, "revenue", control="control", interact=["spend"])
        report = result.model_dump()
        assert report["n"] == _N_CLUSTERS * _PER_CLUSTER
        assert report["n_clusters"] == _N_CLUSTERS
        assert report["intervention_grain"] == "cluster"

    @pytest.mark.slow
    @pytest.mark.parametrize("weighting", ["member_count", "equal"])
    def test_observational_design_preserves_mixed_treatment_dependence_clusters(self, weighting):
        from increment import ClusterBootstrap
        from increment.semantics.design import AdjustmentSet, Observational

        src = _cluster_source(
            pure_arms=False,
            design=Observational(
                control_group="control", adjustment=AdjustmentSet(covariates=("spend",))
            ),
        )
        options: dict = {
            "control": "control",
            "interact": ["spend"],
            "cluster_weight": weighting,
            "bootstrap": ClusterBootstrap(seed=57, repetitions=99),
        }
        result = validate_cate(
            src, "revenue", control="control", interact=["spend"], cluster_weight=weighting
        )
        assert isinstance(result, CateValidation)
        # The honest split stays cluster-atomic although this observational
        # design does not enforce arm purity: a crc32 split of 100 clusters must
        # hold out some, but not all, clusters; all would mean it ignored clusters.
        assert result.n_clusters is not None
        assert 0 < result.n_clusters < _N_CLUSTERS
        assert result.cluster_weight == weighting
        assert result.holdout_ate_se is not None and result.holdout_ate_se > 0
        rule = targeting_rule(src, "revenue", fraction=0.5, **options)
        selected = select_targeting_rule(
            src, "revenue", fractions=(0.5,), n_folds=2, seed=12, **options
        )
        table = _cluster_frame(pure_arms=False)
        cols = {"spend": np.asarray(table["spend"])}
        ids = np.asarray(table["store_id"])
        for policy in (rule, selected.rule):
            assert policy.intervention_grain == policy.deploy_grain == "unit"
            assert policy.cluster_weight == weighting
            assert policy.n_clusters is not None
            actions = policy.predict(cols, cluster_ids=ids)
            assert any(np.unique(actions[ids == label]).size == 2 for label in np.unique(ids))
            _assert_policy_roundtrip(policy, cols, ids)

    def test_unclustered_validation_reports_no_cluster_count(self):
        result = validate_cate(_source(), "revenue", control="control", interact=["spend"])
        assert result.n_clusters is None

    def test_targeting_rule_carries_cluster_count_grain_and_required_columns(self):
        src = _cluster_source(pure_arms=True, intervention_grain="cluster")
        rule = targeting_rule(src, "revenue", control="control", interact=["spend"], fraction=0.3)
        assert rule.n_clusters == rule.validation.n_clusters
        assert rule.n_clusters == rule.validation.n_holdout // _PER_CLUSTER
        assert rule.intervention_grain == "cluster"
        assert rule.required_columns == ("spend",)

    def test_unclustered_targeting_rule_reports_unit_grain_and_no_cluster_count(self):
        rule = targeting_rule(
            _source(), "revenue", control="control", interact=["spend"], fraction=0.3
        )
        assert rule.n_clusters is None
        assert rule.intervention_grain == "unit"
        assert rule.required_columns == ("spend",)

    def test_observational_adjustment_cannot_alias_the_cluster_column(self):
        from increment.semantics.design import AdjustmentSet, Observational

        src = _cluster_source(
            pure_arms=False,
            design=Observational(
                control_group="control", adjustment=AdjustmentSet(covariates=("store_id",))
            ),
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate(src, "revenue", control="control", interact=["spend"])
        assert exc_info.value.code == "cate.cluster.covariate_conflict"

    def test_intervention_grain_cluster_requires_a_declared_cluster_column(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            _cluster_source(cluster=None, intervention_grain="cluster")
        assert exc_info.value.code == "source.frame.intervention_grain_without_cluster"

    def test_declared_cluster_with_no_served_cluster_id_column_always_refuses(self, monkeypatch):
        """A nonconformant `MomentSource` that declares `context.cluster` but
        serves no `cluster_id` column must refuse unconditionally -- silently
        treating this population as unclustered would make cluster-atomic
        splitting/fold assignment and honest validation silently wrong,
        regardless of the declared intervention grain."""
        src = _cluster_source(pure_arms=True)
        real_unit_frame = src.unit_frame

        def dropped(metric, *, covariates=()):
            native = real_unit_frame(metric, covariates=covariates)
            table = native.to_pandas() if hasattr(native, "to_pandas") else native
            return table.drop(columns=["cluster_id"])

        monkeypatch.setattr(src, "unit_frame", dropped)
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_cate(src, "revenue", control="control", interact=["spend"])
        assert exc_info.value.code == "cate.cluster.identity_unavailable"
        assert exc_info.value.context["column"] == "store_id"

    def test_intervention_grain_unavailable_refuses_a_contradictory_raw_source(self, monkeypatch):
        from dataclasses import replace

        source = _source()
        monkeypatch.setattr(
            source, "_context", replace(source.context, intervention_grain="cluster")
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_cate(source, "revenue", control="control", interact=["spend"])
        assert exc_info.value.code == "cate.cluster.intervention_grain_unavailable"

    @pytest.mark.slow
    @pytest.mark.parametrize("store_mode", ["none", "always"])
    @pytest.mark.parametrize(
        "cluster_column, cluster_weight",
        [("store_id", "member_count"), ("store_id", "equal"), (None, "member_count")],
    )
    def test_frame_native_and_artifact_cate_results_preserve_cluster_provenance(
        self, tmp_path, store_mode, cluster_column, cluster_weight
    ):
        """Every source retains the same declared cluster design and fitted result."""
        from datetime import datetime

        import ibis

        from increment.plan import compile_decision_plan
        from increment.query.artifact_publish import artifact_context
        from increment.query.native_source import DefinitionsMomentSource
        from increment.query.session import WarehouseArtifactStore, WarehouseSession
        from increment.query.source import open_artifact
        from increment.semantics.artifact import ClusterIdentityRequest
        from increment.semantics.design import Randomized
        from increment.semantics.loader import load

        frame = _cluster_frame()
        grain = "unit" if cluster_column is None else "cluster"
        expected_clusters = None if cluster_column is None else _N_CLUSTERS
        rows = frame.to_pylist()
        defs_template = """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM cl_events
    timestamp_column: event_at
    entities: [user_id]
    facts:
      - name: exposure
        column: null
      - name: purchase
        column: revenue
exposures:
  - name: enrolled
    fact: exposure
metrics:
  - name: revenue_per_user
    type: mean
    entity: user_id
    fact: purchase
    aggregation: sum
    window_days: 7
experiments:
  - name: store_test
    exposure: enrolled
    unit: user_id
    cluster: store_id
    intervention_grain: cluster
    start: 2025-08-01
    end: 2025-08-07
    plan: {secondaries: [revenue_per_user]}
    control_group: control
"""
        if cluster_column is None:
            defs_template = defs_template.replace("    cluster: store_id\n", "").replace(
                "    intervention_grain: cluster\n", ""
            )
        events = []
        for row in rows:
            events.append(
                {
                    "user_id": row["user_id"],
                    "event_at": datetime(2025, 8, 1, 9),
                    "event": "exposure",
                    "experiment_id": "store_test",
                    "group_id": row["variant"],
                    "store_id": row["store_id"],
                    "revenue": None,
                }
            )
            events.append(
                {
                    "user_id": row["user_id"],
                    "event_at": datetime(2025, 8, 7, 10),
                    "event": "purchase",
                    "experiment_id": None,
                    "group_id": None,
                    "store_id": None,
                    "revenue": row["revenue"],
                }
            )
        con = ibis.duckdb.connect()
        con.create_table("cl_events", obj=events)
        defs_path = tmp_path / "defs.yaml"
        defs_path.write_text(defs_template)
        definitions = load(defs_path)
        experiment = definitions.experiments[0]
        design = Randomized(control_group="control")
        native_src = DefinitionsMomentSource(
            WarehouseSession(con, definitions),
            experiment,
            definitions.metrics,
            store=store_mode,
            on_mixed_assignment="error",
            design=design,
            plan=compile_decision_plan(
                experiment.plan, definitions.metrics, path="warehouse", design=design
            ),
        )

        frame_src = from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[{"name": "revenue_per_user", "type": "mean", "value_column": "revenue"}],
            cluster=cluster_column,
            design=design,
            intervention_grain=grain,
        )

        native_result = estimate_cate(
            native_src,
            "revenue_per_user",
            control="control",
            interact=[],
            cluster_weight=cluster_weight,
        )
        frame_result = estimate_cate(
            frame_src,
            "revenue_per_user",
            control="control",
            interact=[],
            cluster_weight=cluster_weight,
        )
        assert native_result.ate == pytest.approx(frame_result.ate, rel=1e-9)
        assert native_result.se == pytest.approx(frame_result.se, rel=1e-9)
        assert native_result.n_clusters == frame_result.n_clusters == expected_clusters
        assert native_result.intervention_grain == frame_result.intervention_grain == grain
        context = artifact_context(definitions, experiment, "error")
        store = WarehouseArtifactStore(con, schema_name="cate_artifacts")
        reference = native_src.publish_unit_day_artifact(
            store,
            extensions=[]
            if cluster_column is None
            else [ClusterIdentityRequest(cluster_name="store_id")],
        )
        adopted = open_artifact(store, reference, expected_context=context)
        try:
            artifact_result = estimate_cate(
                adopted,
                "revenue_per_user",
                control="control",
                interact=[],
                cluster_weight=cluster_weight,
            )
            assert artifact_result.ate == pytest.approx(native_result.ate, rel=1e-9)
            assert artifact_result.se == pytest.approx(native_result.se, rel=1e-9)
            assert artifact_result.n_clusters == expected_clusters
            assert artifact_result.dimension == native_result.dimension == frame_result.dimension
            assert (
                artifact_result.reference_df
                == native_result.reference_df
                == (_N - artifact_result.dimension if cluster_column is None else _N_CLUSTERS - 1)
            )
            assert np.asarray(artifact_result.vcov) == pytest.approx(np.asarray(native_result.vcov))
            assert np.asarray(frame_result.vcov) == pytest.approx(np.asarray(native_result.vcov))
            assert artifact_result.se_unadjusted == pytest.approx(native_result.se_unadjusted)
            assert artifact_result.intervention_grain == grain
            reports = [
                _source_parity_reports(
                    source,
                    frame,
                    cluster_weight=cluster_weight,
                    clustered=cluster_column is not None,
                )
                for source in (frame_src, native_src, adopted)
            ]
            for expected, actual in zip(reports[0], reports[1], strict=True):
                _assert_cate_payload_close(expected.model_dump(), actual.model_dump())
            for expected, actual in zip(reports[0], reports[2], strict=True):
                _assert_cate_payload_close(expected.model_dump(), actual.model_dump())
        finally:
            adopted.close()
            native_src.close()
            con.disconnect()


@pytest.fixture(scope="module")
def _cluster_policy_json():
    return targeting_rule(
        _cluster_source(intervention_grain="cluster"),
        "revenue",
        control="control",
        interact=["spend"],
        fraction=0.3,
    ).model_dump(mode="json")


@pytest.fixture
def cluster_policy_payload(_cluster_policy_json):
    # Tests mutate the payload, so each one gets a private copy of the fitted rule.
    return copy.deepcopy(_cluster_policy_json)


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("count", [-1, 0, True, 1.5, "2"])
@pytest.mark.parametrize("model", ["rule", "validation"])
def test_policy_cluster_counts_are_strict_positive_integers(
    cluster_policy_payload, serialized, count, model
):
    import json

    from increment.errors import InvalidRequestError

    payload = cluster_policy_payload if model == "rule" else cluster_policy_payload["validation"]
    cls = TargetingRule if model == "rule" else CateValidation
    payload["n_clusters"] = count
    with pytest.raises(InvalidRequestError) as raised:
        if serialized:
            cls.model_validate_json(json.dumps(payload))
        else:
            cls(**payload)
    expected = (
        "model.field.range"
        if isinstance(count, int) and not isinstance(count, bool)
        else "model.field.type"
    )
    assert raised.value.code == expected


@pytest.mark.parametrize("entrypoint", ["constructor", "mapping", "json", "nested"])
@pytest.mark.parametrize(
    "case", ["too_many", "mismatch", "missing_rule", "missing_validation", "missing_both"]
)
def test_policy_cluster_metadata_rejects_inconsistent_states(
    cluster_policy_payload, entrypoint, case
):
    import json

    payload = cluster_policy_payload
    if case == "too_many":
        payload["n_clusters"] = payload["validation"]["n_holdout"] + 1
        payload["validation"]["n_clusters"] = payload["n_clusters"]
        code = "estimation.targeting.cluster_count_bounds"
        context = {
            "n_clusters": payload["n_clusters"],
            "n_holdout": payload["validation"]["n_holdout"],
        }
    else:
        if case == "mismatch":
            payload["n_clusters"] += 1
        if case in ("missing_rule", "missing_both"):
            payload["n_clusters"] = None
        if case in ("missing_validation", "missing_both"):
            payload["validation"]["n_clusters"] = None
        code = "estimation.targeting.cluster_count_mismatch"
        context = {
            "n_clusters": payload["n_clusters"],
            "validation_n_clusters": payload["validation"]["n_clusters"],
        }
        if case == "missing_both":
            code = "estimation.targeting.cluster_grain_count_required"
            context = {}
    with pytest.raises(InvalidRequestError) as caught:
        if entrypoint == "constructor":
            TargetingRule(**payload)
        elif entrypoint == "mapping":
            TargetingRule.model_validate(payload)
        elif entrypoint == "json":
            TargetingRule.model_validate_json(json.dumps(payload))
        else:
            TargetingSelection.model_validate_json(
                json.dumps(
                    {
                        "fractions": [payload["fraction"]],
                        "inner": [
                            {
                                "fraction": payload["fraction"],
                                "achieved_fraction": payload["achieved_fraction"],
                                "net_benefit": payload["validation"]["holdout_ate"],
                                "n": payload["validation"]["n_holdout"],
                            }
                        ],
                        "selected_fraction": payload["fraction"],
                        "cost_per_treated": 0.0,
                        "n_folds": 2,
                        "seed": 0,
                        "rule": payload,
                    }
                )
            )
    assert caught.value.code == code
    assert caught.value.context == context
    with pytest.raises(TypeError):
        cast("dict[str, object]", caught.value.context)["n_clusters"] = 0


@pytest.mark.parametrize("grain, count", [("unit", None), ("unit", 1), ("cluster", 1)])
def test_policy_cluster_metadata_round_trips_valid_boundaries(cluster_policy_payload, grain, count):
    payload = cluster_policy_payload
    payload["intervention_grain"] = grain
    payload["deploy_grain"] = grain
    payload["budget_rule"] = "whole_cluster_prefix" if grain == "cluster" else "unit_threshold"
    if grain == "unit":
        payload["score_cutoff"] = 0.0
    payload["n_clusters"] = payload["validation"]["n_clusters"] = count
    payload["validation"]["n_holdout"] = 1
    rule = TargetingRule(**payload)
    assert TargetingRule.model_validate_json(rule.model_dump_json()) == rule
    assert rule.n_clusters == rule.validation.n_clusters == count


def _cluster_oracle_source(*, intervention_grain="unit"):
    sizes = np.array([2, 4, 2, 4, 2, 4, 4, 2])
    table = pa.table(
        {
            "user_id": np.arange(24),
            "variant": np.repeat(["control"] * 4 + ["treatment"] * 4, sizes),
            "revenue": np.repeat([0.0, 2.0, 0.0, 4.0, 1.0, 7.0, 4.0, 12.0], sizes),
            "segment": np.repeat(["A", "A", "B", "B", "A", "A", "B", "B"], sizes),
            "store_id": np.repeat([f"cluster-{i}" for i in range(8)], sizes),
        }
    )
    return from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        cluster="store_id",
        intervention_grain=intervention_grain,
    )


@pytest.mark.parametrize("grain", ["unit", "cluster"])
def test_public_cluster_weighting_is_explicit_and_uses_independent_oracle(grain):
    from scipy.stats import t

    source = _cluster_oracle_source(intervention_grain=grain)
    interact = [Covariate(name="segment", kind="categorical")]
    default = estimate_cate(source, "revenue", control="control", interact=interact)
    member = estimate_cate(
        source, "revenue", control="control", interact=interact, cluster_weight="member_count"
    )
    equal = estimate_cate(
        source, "revenue", control="control", interact=interact, cluster_weight="equal"
    )
    assert default.model_dump() == member.model_dump()
    assert member.ate == pytest.approx(23 / 6)
    assert equal.ate == pytest.approx(4.5)
    assert equal.se**2 == pytest.approx(30 / 7)
    for result, expected_contrast, expected_variance in [
        (member, 1 / 3, 2560 / 189),
        (equal, 3, 120 / 7),
    ]:
        contrast = result.contrast({"segment": "B"}, {"segment": "A"})
        assert contrast.value == pytest.approx(expected_contrast)
        assert contrast.ub is not None
        assert contrast.ub - contrast.value == pytest.approx(
            t.isf(0.025, 7) * np.sqrt(expected_variance)
        )
        assert (result.dimension, result.n_clusters, result.reference_df) == (4, 8, 7)
        report = result.model_dump()
        assert report["dimension"] == 4
        assert report["n_clusters"] == 8
        assert report["reference_df"] == 7
        assert report["cluster_weight"] == result.cluster_weight
        assert result.unadjusted_vcov is not None
        assert report["se_unadjusted"] == pytest.approx(np.sqrt(result.unadjusted_vcov[1][1]))


def test_public_equal_cluster_weighting_refuses_before_reading_unclustered_source(monkeypatch):
    source = _source()

    def unavailable_frame(*args, **kwargs):
        raise AssertionError("invalid weighting must refuse before requesting rows")

    monkeypatch.setattr(source, "unit_frame", unavailable_frame)
    with pytest.raises(InvalidRequestError) as caught:
        estimate_cate(source, "revenue", control="control", interact=[], cluster_weight="equal")
    assert caught.value.code == "estimation.cate.equal_weighting_without_cluster"


@pytest.mark.parametrize("caller", ["validate", "rule", "selection"])
def test_public_validation_rechecks_forged_bootstrap_before_loading_units(caller, monkeypatch):
    from increment import ClusterBootstrap

    source = _source()
    options = ClusterBootstrap().model_copy(update={"seed": True})

    def reject_load(*args, **kwargs):
        pytest.fail("invalid bootstrap options must refuse before loading unit data")

    monkeypatch.setattr(type(source), "unit_frame", reject_load)
    with pytest.raises(InvalidRequestError) as error:
        if caller == "validate":
            validate_cate(
                source, "revenue", control="control", interact=["spend"], bootstrap=options
            )
        elif caller == "rule":
            targeting_rule(
                source,
                "revenue",
                control="control",
                interact=["spend"],
                fraction=0.5,
                bootstrap=options,
            )
        else:
            select_targeting_rule(
                source,
                "revenue",
                control="control",
                interact=["spend"],
                fractions=(0.5,),
                seed=0,
                bootstrap=options,
            )
    assert error.value.code == "estimation.targeting.bootstrap_options"


@pytest.mark.parametrize("options", [{"seed": True}, {"seed": 0.5}, {"repetitions": 1}])
def test_bootstrap_options_reject_invalid_integer_controls(options):
    from increment import ClusterBootstrap

    with pytest.raises(InvalidRequestError) as error:
        ClusterBootstrap.model_validate(options)
    assert error.value.code == "estimation.targeting.bootstrap_options"


@pytest.mark.parametrize("caller", [targeting_rule, select_targeting_rule])
def test_public_incompatible_deployment_refuses_before_loading_units(caller, monkeypatch):
    source = _cluster_source(intervention_grain="cluster")

    def no_units(*args, **kwargs):
        pytest.fail("incompatible deployment must refuse before unit-frame loading")

    monkeypatch.setattr(type(source), "unit_frame", no_units)
    options: dict = (
        {"fraction": 0.5} if caller is targeting_rule else {"fractions": (0.5,), "seed": 1}
    )
    with pytest.raises(InvalidRequestError) as caught:
        caller(
            source, "revenue", control="control", interact=["spend"], deploy_grain="unit", **options
        )
    assert caught.value.code == "estimation.targeting.unsupported_unit_deployment"


@pytest.mark.slow
@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_analysis_delegates_preserve_cluster_intervention_and_portable_prediction(weighting):
    from increment import Analysis, ClusterBootstrap
    from increment.results import ClusterScore

    table = _cluster_frame()
    analysis = Analysis.from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
        cluster="store_id",
        intervention_grain="cluster",
    )
    ids = np.asarray(table["store_id"].to_numpy())
    cols = {"spend": np.asarray(table["spend"].to_numpy())}
    fit = analysis.estimate_cate(
        "revenue", control="control", interact=["spend"], cluster_weight=weighting
    )
    assert fit.cluster_weight == weighting
    scores: tuple[ClusterScore, ...] = fit.score(cols, cluster_ids=ids, deploy_grain="cluster")
    assert len(scores) == _N_CLUSTERS
    bootstrap = ClusterBootstrap(seed=12, repetitions=9)
    validation = analysis.validate_cate(
        "revenue",
        control="control",
        interact=["spend"],
        bootstrap=bootstrap,
        cluster_weight=weighting,
    )
    assert validation.n_clusters is not None
    assert validation.cluster_weight == weighting
    rule = analysis.targeting_rule(
        "revenue",
        control="control",
        interact=["spend"],
        fraction=0.5,
        cluster_weight=weighting,
        bootstrap=bootstrap,
    )
    selection = analysis.select_targeting_rule(
        "revenue",
        control="control",
        interact=["spend"],
        fractions=(0.5,),
        seed=12,
        n_folds=2,
        cluster_weight=weighting,
        bootstrap=bootstrap,
    )
    for policy in (rule, selection.rule):
        assert policy.intervention_grain == policy.deploy_grain == "cluster"
        assert policy.cluster_weight == weighting
        assert (policy.bootstrap_seed, policy.bootstrap_repetitions) == (12, 9)
        actions = policy.predict(cols, cluster_ids=ids)
        for cluster in np.unique(ids):
            assert np.unique(actions[ids == cluster]).size == 1
        restored = TargetingRule.model_validate_json(policy.model_dump_json())
        np.testing.assert_array_equal(restored.predict(cols, cluster_ids=ids), actions)


def _c01_pattern_source(replication):
    from tests.estimation.test_targeting import _c01_five_row_pattern

    data = _c01_five_row_pattern(replication)
    data["cols"]["segment"] = np.where(data["cols"]["spend"] > 0, "high", "low")
    table = pa.table(
        {
            "unit": data["unit_ids"],
            "cluster": data["cluster_ids"],
            "arm": np.where(data["d"] == 0, "control", "treatment"),
            "y": data["y"],
            **data["cols"],
        }
    )
    source = from_unit_summary(
        table,
        unit="unit",
        group="arm",
        control="control",
        metrics={"y": "mean"},
        cluster="cluster",
        intervention_grain="cluster",
        design=Randomized(control_group="control"),
    )
    return data, source


def _assert_c01_public_fit_covariance(data, fit, weighting):
    y, d, ids = data["y"], data["d"], data["cluster_ids"]
    labels, inverse, sizes = np.unique(ids, return_inverse=True, return_counts=True)
    weights = np.ones(y.size) if weighting == "member_count" else 1 / sizes[inverse]
    x = data["cols"]["spend"]
    centered = x - np.average(x, weights=weights)
    z = np.column_stack([np.ones(y.size), d, centered, d * centered])
    bread = np.linalg.inv(z.T @ (weights[:, None] * z))
    beta = bread @ (z.T @ (weights * y))
    residual = y - z @ beta
    scores = np.stack(
        [
            np.sum((weights * residual)[:, None][ids == label] * z[ids == label], axis=0)
            for label in labels
        ]
    )
    covariance = 40 / 39 * bread @ (scores.T @ scores) @ bread
    assert fit.ate == pytest.approx(beta[1])
    assert fit.se**2 == pytest.approx(covariance[1, 1])


@pytest.mark.slow
@pytest.mark.parametrize("weighting", ["member_count", "equal"])
@pytest.mark.parametrize("replication", [1, 5, 20, 100])
def test_cluster_complete_five_row_pattern_full_public_pipeline(weighting, replication):
    from increment import ClusterBootstrap
    from increment.estimation.crossfit import outer_split

    data, source = _c01_pattern_source(replication)
    d, ids = data["d"], data["cluster_ids"]
    options: dict = {"control": "control", "interact": ["spend"], "cluster_weight": weighting}
    fit = estimate_cate(source, "y", **options)
    shrunk = estimate_cate(source, "y", ard=True, **options)
    _assert_c01_public_fit_covariance(data, fit, weighting)
    assert fit.n_clusters == shrunk.n_clusters == 40
    assert fit.reference_df == shrunk.reference_df == 39
    np.testing.assert_allclose(shrunk.vcov, fit.vcov)
    assert shrunk.interactions[0].ard_coef is not None
    assert shrunk.cate({"spend": 1.0}).lb is None
    assert fit.cate({"spend": 1.0}).lb is not None
    assert np.isfinite(shrunk.score_state.score(data["cols"])).all()

    # Each fit relearns its continuous basis; no equality of refitted p-values is asserted.
    bootstrap = ClusterBootstrap(seed=57, repetitions=99)
    validation = validate_cate(
        source,
        "y",
        n_groups=2,
        bootstrap=bootstrap,
        include_evaluation_population=True,
        **options,
    )
    held = validation.evaluation_population
    assert held is not None and held.split == "honest"
    hold = np.isin(data["unit_ids"], np.asarray(held.unit_ids))
    assert not set(ids[hold]) & set(ids[~hold])
    assert validation.n_train == int((~hold).sum())
    assert validation.n_holdout == int(hold.sum())
    assert validation.n_clusters == np.unique(ids[hold]).size
    assert validation.holdout_ate_se is not None and validation.holdout_ate_se > 0
    assert validation.autoc.se is not None and validation.autoc.se > 0
    assert validation.qini.se is not None and validation.qini.se > 0
    rule = targeting_rule(source, "y", fraction=0.5, bootstrap=bootstrap, **options)
    selection = select_targeting_rule(
        source,
        "y",
        fractions=(0.0, 0.5, 1.0),
        n_folds=2,
        seed=12,
        bootstrap=bootstrap,
        **options,
    )
    outer = outer_split(data["unit_ids"], test_size=0.5, seed=12, stratify=d, cluster_ids=ids)
    assert not set(ids[outer]) & set(ids[~outer])
    assert selection.rule.validation.n_clusters == np.unique(ids[outer]).size
    assert selection.n_clusters == np.unique(ids[~outer]).size
    for policy in (rule, selection.rule):
        assert policy.cluster_weight == weighting
        assert policy.intervention_grain == policy.deploy_grain == "cluster"
        assert policy.achieved_fraction <= policy.fraction
        _assert_policy_roundtrip(policy, data["cols"], ids)


@pytest.mark.slow
@pytest.mark.parametrize("weighting", ["member_count", "equal"])
@pytest.mark.parametrize("replication", [1, 5, 20, 100])
def test_public_cluster_complete_pattern_ard_precision(weighting, replication):
    """A fixed categorical basis isolates ARD precision from relearned scaling."""
    base, original_source = _c01_pattern_source(1)
    _, repeated_source = _c01_pattern_source(replication)
    options: dict = {
        "control": "control",
        "interact": [Covariate(name="segment", kind="categorical")],
        "ard": True,
        "cluster_weight": weighting,
    }
    original = estimate_cate(original_source, "y", **options)
    repeated = estimate_cate(repeated_source, "y", **options)
    assert original.n_clusters == repeated.n_clusters == 40
    assert original.reference_df == repeated.reference_df == 39
    assert original.beta_ard is not None and repeated.beta_ard is not None
    np.testing.assert_allclose(repeated.beta, original.beta, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(repeated.vcov, original.vcov, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(repeated.beta_ard, original.beta_ard, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        repeated.score_state.score(base["cols"]),
        original.score_state.score(base["cols"]),
        rtol=1e-9,
        atol=1e-10,
    )
    assert repeated.se_unadjusted == pytest.approx(original.se_unadjusted)
