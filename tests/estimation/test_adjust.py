"""Tests for IPTW scoring, the identification gate, and `estimate_ate`."""

from __future__ import annotations

import re
import warnings
from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pytest
from scipy.special import expit

from increment import Analysis, IdentificationError
from increment.errors import (
    CapabilityError,
    IncrementRuntimeWarning,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation._adjust.common import _refuse_near_zero_adjustment_denominator
from increment.estimation._adjust.dml import _dml_theta
from increment.estimation._adjust.learners import LogisticPropensity, RidgeOutcome
from increment.estimation.adjust import estimate_ate
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.frame import MetricSpec, from_unit_summary
from increment.semantics.design import AdjustmentSet, IdentificationGate, Observational
from tests.analysis_factory import lift_rows
from tests.categorical_cases import (
    DUMMY_COLUMNS,
    assert_rows_match,
    categorical_units,
    dummy_table,
    raw_table,
    with_nulls,
)
from tests.warning_codes import warning_codes


class StubLearner:
    """Deterministic Learner: returns preset propensities. Also proves the
    protocol seam accepts user objects (no inheritance)."""

    def __init__(self, e):
        self._e = np.asarray(e, dtype=float)

    def fit(self, X, d):
        pass

    def predict(self, X):
        return self._e


def test_adjust_public_surface_keeps_only_dispatch_policy():
    import increment.estimation.adjust as adjust_module

    assert callable(adjust_module.estimate_ate)
    assert callable(adjust_module.judge_shared_prior_scales)
    assert not hasattr(adjust_module, "iptw_estimate")
    assert not hasattr(adjust_module, "dml_estimate")
    assert not hasattr(adjust_module, "aipw_estimate")


def _src():  # 6 units, covariate z constant (=> SMD 0, no warning; zero-var guard)
    tbl = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "variant": ["T", "T", "T", "C", "C", "C"],
            "revenue": [3.0, 5.0, 4.0, 1.0, 2.0, 3.0],
            "z": [1.0] * 6,
        }
    )
    return from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )


def _src_without_u1():  # the rows _src() keeps once the e=0.999 unit is trimmed
    tbl = pa.table(
        {
            "user_id": ["u2", "u3", "u4", "u5", "u6"],
            "variant": ["T", "T", "C", "C", "C"],
            "revenue": [5.0, 4.0, 1.0, 2.0, 3.0],
            "z": [1.0] * 5,
        }
    )
    return from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )


def _winsor_src():
    tbl = pa.table(
        {
            "user_id": [f"u{i}" for i in range(12)],
            "variant": ["T"] * 6 + ["C"] * 6,
            "revenue": [
                3.0,
                5.0,
                4.0,
                3.0,
                4.0,
                5.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                3.0,
            ],
            "z": [1.0] * 12,
        }
    )
    return from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[
            MetricSpec(
                name="revenue",
                winsorization={"upper_value": 4.5},
            )
        ],
    )


def _percentile_winsor_src():
    tbl = pa.table(
        {
            "user_id": [f"u{i}" for i in range(12)],
            "variant": ["T"] * 6 + ["C"] * 6,
            "revenue": [3.0, 5.0, 4.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "z": [1.0] * 12,
        }
    )
    return from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[
            MetricSpec(
                name="revenue",
                winsorization={"upper_percentile": 0.99},
            )
        ],
    )


_DESIGN = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("z",)))


def _metric(src, name):
    """The Metric object the source declared for *name* (src.metrics is the
    authoritative list - same lookup readouts.run performs)."""
    return next(m for m in src.context.metrics if m.name == name)


def test_iptw_hand_computed_hajek_point_and_se():
    # Fixed propensity Hajek means are 4 and 13/7. Independent arm score
    # contributions give control variance15872/(49²*36), treatment512/(49*36).
    def stub():
        return StubLearner([0.5, 0.5, 0.8, 0.5, 0.5, 0.2])

    (est,) = estimate_ate(
        _src(), _DESIGN, methods=[Method(name="iptw", propensity_learner=stub)]
    ).results
    var_c = 15872 / (49**2 * 36)
    var_t = 512 / (49 * 36)
    assert est.method == "iptw" and est.group_id == "T"
    assert est.require_lift().value == pytest.approx(15 / 13, rel=1e-12)
    assert est.relative_confidence_set is not None
    reference = est.relative_confidence_set.reference
    assert reference.var_a == pytest.approx(var_t + var_c, rel=1e-12)
    assert reference.var_c == pytest.approx(var_c, rel=1e-12)
    assert reference.cov_ac == pytest.approx(-var_c, rel=1e-12)
    assert est.population is None


def test_control_absent_raises_and_lists_available_groups():
    def stub():
        return StubLearner([0.5, 0.5, 0.8, 0.5, 0.5, 0.2])

    design = Observational(control_group="ZZ", adjustment=AdjustmentSet(covariates=("z",)))
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_ate(_src(), design, methods=[Method(name="iptw", propensity_learner=stub)])
    assert exc_info.value.code == "estimation.adjust_common.control_group_found"


def test_overlap_refuses_by_default_and_names_trim():
    def stub():
        return StubLearner([0.999, 0.5, 0.8, 0.5, 0.5, 0.2])

    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(_src(), _DESIGN, methods=[Method(name="iptw", propensity_learner=stub)])
    assert exc_info.value.code == "adjust.identification.overlap_gate_refused"


def test_trim_records_population_and_drops_units():
    def stub():
        return StubLearner([0.999, 0.5, 0.8, 0.5, 0.5, 0.2])

    d = _DESIGN.model_copy(update={"gate": IdentificationGate(overlap="trim")})
    (est,) = estimate_ate(_src(), d, methods=[Method(name="iptw", propensity_learner=stub)]).results
    assert est.population is not None

    # The e=0.999 unit is excluded: the trimmed estimate is the untrimmed one
    # computed over exactly the 5 units inside [0.01, 0.99].
    def stub_retained():
        return StubLearner([0.5, 0.8, 0.5, 0.5, 0.2])

    (untrimmed,) = estimate_ate(
        _src_without_u1(),
        _DESIGN,
        methods=[Method(name="iptw", propensity_learner=stub_retained)],
    ).results
    assert est.require_lift().value == pytest.approx(untrimmed.require_lift().value, rel=1e-12)
    assert est.relative_confidence_set is not None
    assert untrimmed.relative_confidence_set is not None
    reference, reference_untrimmed = (
        est.relative_confidence_set.reference,
        untrimmed.relative_confidence_set.reference,
    )
    assert reference.var_a == pytest.approx(reference_untrimmed.var_a, rel=1e-12)
    assert reference.var_c == pytest.approx(reference_untrimmed.var_c, rel=1e-12)
    assert reference.cov_ac == pytest.approx(reference_untrimmed.cov_ac, rel=1e-12)


# Observational rows must stamp the estimand they actually identify
# (ate / plr_slope / overlap_subpopulation_ate), never the randomized/
# encouragement default "itt" - see infer_ate.


def test_iptw_estimand_is_ate_without_trim():
    def stub():
        return StubLearner([0.5, 0.5, 0.8, 0.5, 0.5, 0.2])

    (est,) = estimate_ate(
        _src(), _DESIGN, methods=[Method(name="iptw", propensity_learner=stub)]
    ).results
    assert est.estimand == "ate"


def test_iptw_estimand_is_overlap_subpopulation_ate_when_trimmed():
    def stub():
        return StubLearner([0.999, 0.5, 0.8, 0.5, 0.5, 0.2])

    d = _DESIGN.model_copy(update={"gate": IdentificationGate(overlap="trim")})
    (est,) = estimate_ate(_src(), d, methods=[Method(name="iptw", propensity_learner=stub)]).results
    assert est.estimand == "overlap_subpopulation_ate"


def test_trim_emptying_an_arm_refuses():
    def stub():
        return StubLearner([0.999, 0.999, 0.999, 0.5, 0.5, 0.2])

    d = _DESIGN.model_copy(update={"gate": IdentificationGate(overlap="trim")})
    with pytest.raises(IdentificationError):
        estimate_ate(_src(), d, methods=[Method(name="iptw", propensity_learner=stub)])


def test_max_smd_gate_refuses_on_imbalance():
    # Covariate x tracks d perfectly; stub propensities keep overlap fine, so
    # only the balance gate can fire.
    tbl = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "variant": ["T", "T", "T", "C", "C", "C"],
            "revenue": [3.0, 5.0, 4.0, 1.0, 2.0, 3.0],
            "x": [2.0, 2.1, 1.9, 0.1, 0.0, -0.1],
        }
    )
    src = from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x",)),
        gate=IdentificationGate(max_smd=0.05),
    )

    def stub():
        return StubLearner([0.6, 0.6, 0.6, 0.4, 0.4, 0.4])

    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, design, methods=[Method(name="iptw", propensity_learner=stub)])
    assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"


def test_zero_within_arm_variance_refuses_complete_separation():
    """A covariate constant inside each arm but different between them is
    COMPLETE separation -- the worst imbalance possible. The pooled sd is 0, and
    returning 0.0 for that (as a divide-by-zero guard once did) reported the
    worst case as perfect balance and silently disabled the gate the caller
    configured. Stub propensities keep overlap fine so only the balance gate can
    fire."""
    tbl = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "variant": ["T", "T", "T", "C", "C", "C"],
            "revenue": [3.0, 5.0, 4.0, 1.0, 2.0, 3.0],
            "x": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    src = from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x",)),
        gate=IdentificationGate(max_smd=0.05),
    )

    def stub():
        return StubLearner([0.6, 0.6, 0.6, 0.4, 0.4, 0.4])

    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, design, methods=[Method(name="iptw", propensity_learner=stub)])
    assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"


def test_a_constant_covariate_in_both_arms_still_passes_the_balance_gate():
    """The other side of the zero-pooled-sd boundary: a covariate constant at the
    SAME value in both arms has no imbalance to report, so it must not be refused.
    An awkward value like 0.1, whose weighted arm means agree only up to the
    rounding of their weighted sums, must still read as balanced."""
    tbl = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "variant": ["T", "T", "T", "C", "C", "C"],
            "revenue": [3.0, 5.0, 4.0, 1.0, 2.0, 3.0],
            "x": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )
    src = from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x",)),
        gate=IdentificationGate(max_smd=0.05),
    )

    def stub():
        return StubLearner([0.6, 0.6, 0.6, 0.4, 0.4, 0.4])

    (est,) = estimate_ate(
        src, design, methods=[Method(name="iptw", propensity_learner=stub)]
    ).results
    assert est.require_lift().value == pytest.approx(1.0)


def test_large_close_but_distinct_arm_levels_still_refuse():
    """A relative tolerance on the weighted arm means is not enough. At a
    magnitude of 1e9 a half-unit gap is under 1e-9 relatively, so comparing the
    means with any relative tolerance equates two DISTINCT constant levels and
    lets complete separation past the gate again. Equality is therefore decided
    on the raw covariate levels, which are exact."""
    tbl = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "variant": ["T", "T", "T", "C", "C", "C"],
            "revenue": [3.0, 5.0, 4.0, 1.0, 2.0, 3.0],
            "x": [1e9, 1e9, 1e9, 1e9 + 0.5, 1e9 + 0.5, 1e9 + 0.5],
        }
    )
    src = from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x",)),
        gate=IdentificationGate(max_smd=0.05),
    )

    def stub():
        return StubLearner([0.6, 0.6, 0.6, 0.4, 0.4, 0.4])

    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, design, methods=[Method(name="iptw", propensity_learner=stub)])
    assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"


def test_max_smd_refuses_inside_the_strict_band_below_point_one():
    """Regression: max_smd must gate on abs(smd) > max_smd directly, NOT on
    a hardcoded > 0.1 pre-filter. A weighted SMD of ~0.06 with max_smd=0.03
    must refuse even though 0.06 is well below the 0.1 advisory threshold;
    the bug this guards against silently passed any SMD in (max_smd, 0.1]."""
    treated_x = [-2.4488, -1.4488, -0.4488, 0.5512, 1.5512, 2.5512]
    control_x = [-2.5512, -1.5512, -0.5512, 0.4488, 1.4488, 2.4488]
    tbl = pa.table(
        {
            "user_id": [f"u{i}" for i in range(12)],
            "variant": ["T"] * 6 + ["C"] * 6,
            "revenue": [10.0 + v for v in treated_x + control_x],
            "x": treated_x + control_x,
        }
    )
    src = from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x",)),
        gate=IdentificationGate(max_smd=0.03),
    )

    def stub():  # uniform propensity -> weighted SMD == plain SMD
        return StubLearner([0.5] * 12)

    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, design, methods=[Method(name="iptw", propensity_learner=stub)])
    assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"


def test_max_smd_gate_denominator_is_unweighted_pooled_sd():
    """The balance SMD's pooled sd must come from each arm's own
    (unweighted) sample variance (Austin & Stuart 2015), not a weighted
    pooled sd -- a weighted denominator is a function of the very weights
    the gate is diagnosing, so an inflating weight scheme can shrink it
    enough to slip an imbalanced covariate past `gate.max_smd`.

    Two control units sit at the overlap boundary (e=0.99, ~100x a
    majority unit's weight), placed symmetrically around the SAME
    control-arm mean as the other four -- the weighted MEAN is unaffected,
    but a weighted pooled sd denominator would balloon (hand-computed
    weighted-sd SMD ~=-0.28; unweighted-sd SMD ~=-0.40). At
    gate.max_smd=0.35 the buggy denominator would silently pass this
    covariate at the exact extreme-weight regime the gate exists to catch;
    the corrected denominator must refuse it.
    """
    treated_x = [0.5, 1.0, 1.5, 0.75, 1.25, 1.0]
    control_x = [0.8, 1.3, 1.8, 1.3, -0.2, 2.8]
    tbl = pa.table(
        {
            "user_id": [f"u{i}" for i in range(12)],
            "variant": ["T"] * 6 + ["C"] * 6,
            "revenue": [10.0 + v for v in treated_x + control_x],
            "x": treated_x + control_x,
        }
    )
    src = from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x",)),
        gate=IdentificationGate(max_smd=0.35),
    )

    def stub():
        return StubLearner([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.99, 0.99])

    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, design, methods=[Method(name="iptw", propensity_learner=stub)])
    assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"


def test_advisory_smd_warns_over_point_one():
    # Same imbalance, but max_smd unset -> advisory UserWarning, estimate returned.
    tbl = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "variant": ["T", "T", "T", "C", "C", "C"],
            "revenue": [3.0, 5.0, 4.0, 1.0, 2.0, 3.0],
            "x": [2.0, 2.1, 1.9, 0.1, 0.0, -0.1],
        }
    )
    src = from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))

    def stub():
        return StubLearner([0.6, 0.6, 0.6, 0.4, 0.4, 0.4])

    with pytest.warns(IncrementWarning) as rec:
        (est,) = estimate_ate(
            src, design, methods=[Method(name="iptw", propensity_learner=stub)]
        ).results
    assert "estimation.adjust_common.covariate_balance_advisory" in warning_codes(rec)
    assert est.method == "iptw"


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.adjust_dml.dml_residualized_treatment",
            lambda: _dml_theta(
                np.zeros(2),
                np.array([1.0, 2.0]),
                metric_name="m",
                contrast="c",
            ),
        ),
        (
            "estimation.adjust_common.statistically_indistinguishable_from",
            lambda: _refuse_near_zero_adjustment_denominator(
                SimpleNamespace(  # ty: ignore[invalid-argument-type]
                    metric=SimpleNamespace(name="m"),
                    treatment_group="t",
                    control_group="c",
                    method="AIPW",
                    prior=Normal(mu=0.0, sigma=1.0),
                ),
                0.0,
                1.0,
                label="tau",
            ),
        ),
    ],
)
def test_adjust_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code


def test_near_zero_denominator_refusal_carries_the_refusing_cells_diagnostics():
    """The refusal must name which cell failed (estimator/metric/arms) and
    carry the numbers that triggered it (denominator/SE), not just a
    generic message -- otherwise a multi-arm/multi-metric call gives no
    way to tell which row refused."""
    with pytest.raises(InvalidRequestError) as exc_info:
        _refuse_near_zero_adjustment_denominator(
            SimpleNamespace(  # ty: ignore[invalid-argument-type]
                metric=SimpleNamespace(name="m"),
                treatment_group="t",
                control_group="c",
                method="AIPW",
                prior=Normal(mu=0.0, sigma=1.0),
            ),
            0.0,
            1.0,
            label="tau",
        )
    context = exc_info.value.context
    assert context["label"] == "tau"
    assert context["denominator"] == 0.0
    assert context["se"] == 1.0
    assert context["metric"] == "m"
    assert context["treatment"] == "t"
    assert context["control"] == "c"
    assert context["estimator"] == "AIPW"


def test_estimate_ate_defaults_to_iptw_method():
    # methods=None must label results "iptw". No stub needed: the sole
    # covariate is constant, so overlap trivially passes.
    ests = estimate_ate(_src(), _DESIGN).results
    assert [e.method for e in ests] == ["iptw"]


def test_estimate_ate_duplicate_method_names_refuse_before_source_access(monkeypatch):
    src = _src()

    def fail(*_args, **_kwargs):
        raise AssertionError("source moments accessed before validation")

    monkeypatch.setattr(src, "moments", fail)
    with pytest.raises(InvalidRequestError) as raised:
        estimate_ate(
            src,
            _DESIGN,
            methods=[Method(name="same"), Method(name="same", variance_reduction="cuped")],
        )
    assert raised.value.code == "estimation.engine.method_names_unique"


def test_estimate_ate_mixed_prior_method_scales_refuse_before_source_access(monkeypatch):
    src = _src()

    def fail(*_args, **_kwargs):
        raise AssertionError("source moments accessed before validation")

    monkeypatch.setattr(src, "moments", fail)
    with pytest.raises(InvalidRequestError) as raised:
        estimate_ate(
            src,
            _DESIGN,
            methods=[Method(name="unadjusted"), Method(name="iptw")],
            prior=Normal(mu=0.0, sigma=0.05),
        )

    error = raised.value
    assert error.code == "estimation.adjust.prior.method_scale"
    assert error.context == {}


def test_estimate_ate_mixture_prior_refuses_with_stable_contract():
    from increment.estimation.priors import MixturePrior

    with pytest.raises(InvalidRequestError) as raised:
        estimate_ate(
            _src(),
            _DESIGN,
            prior=MixturePrior(weights=(1.0,), means=(0.0,), sigmas=(0.1,)),
        )

    error = raised.value
    assert error.code == "estimation.adjust.prior.type"
    assert error.context == {}


@pytest.mark.parametrize("method_name", ["iptw", "dml", "aipw"])
def test_fixed_winsorization_diagnostics_reach_adjusted_estimators(method_name):
    if method_name == "iptw":
        method = Method(name=method_name)
    else:
        method = Method(
            name=method_name,
            propensity_learner=LogisticPropensity,
            outcome_learner=RidgeOutcome,
            folds=2,
        )
    (estimate,) = estimate_ate(
        _winsor_src(),
        _DESIGN,
        methods=[method],
    ).results

    assert estimate.winsor_upper_bound == 4.5
    assert estimate.winsor_control_n == 6
    assert estimate.winsor_control_n_upper == 0
    assert estimate.winsor_treatment_n == 6
    assert estimate.winsor_treatment_n_upper == 2


def test_estimate_ate_unadjusted_explicit_optin():
    # Explicit opt-in runs the existing moments path - confounded, labeled.
    ests = estimate_ate(_src(), _DESIGN, methods=[Method(name="unadjusted")]).results
    assert [e.method for e in ests] == ["unadjusted"]
    # Unweighted comparison of raw means: mu_T/mu_C - 1 = 4.0/2.0 - 1 = 1.0
    assert ests[0].require_lift().value == pytest.approx(1.0, rel=1e-2)


def test_estimate_ate_unadjusted_undeclared_metric_preferred_direction_is_none():
    """The unadjusted branch's own `estimate_lift(...)` call
    used to unconditionally stamp `preferred_direction=metric.preferred_direction`,
    silently defaulting an undeclared metric to "increase" -- bypassing
    the readouts' explicitness fix entirely for every Observational-design run."""
    ests = estimate_ate(_src(), _DESIGN, methods=[Method(name="unadjusted")]).results
    assert all(e.preferred_direction is None for e in ests)


def test_estimate_ate_iptw_undeclared_metric_preferred_direction_is_none():
    """The same defect's other half: the adjust_fn branch's `iptw_estimate(...)`
    call had the identical unconditional stamp."""
    ests = estimate_ate(_src(), _DESIGN, methods=[Method(name="iptw")]).results
    assert all(e.preferred_direction is None for e in ests)


def test_estimate_ate_refuses_silently_ignored_variance_reduction():
    """variance_reduction='cuped' under an adjustment method used to be
    silently ignored (bit-identical output to plain iptw): the caller asked
    for a reduction and got none, unlabeled. Refuse by name.

    The refusal is a coded UnsupportedRequestError carrying
    `.code` and the method/variance_reduction context.
    """
    with pytest.raises(UnsupportedRequestError) as excinfo:
        estimate_ate(_src(), _DESIGN, methods=[Method(name="iptw", variance_reduction="cuped")])
    error = excinfo.value
    assert error.code == "adjust.variance_reduction.unsupported"
    assert error.context == {"method": "iptw", "variance_reduction": "cuped"}


def test_estimate_ate_refuses_percentile_winsorized_metric():
    """The generic `capability.error` construction is split
    into a dedicated registered code carrying the offending metric name,
    instead of an unstructured message-only CapabilityError."""
    with pytest.raises(CapabilityError) as excinfo:
        estimate_ate(_percentile_winsor_src(), _DESIGN)
    error = excinfo.value
    assert error.code == "adjust.winsorization.percentile_unsupported"
    assert error.context == {"metric": "revenue"}


def test_estimate_ate_unknown_method_names_available():
    with pytest.raises(UnsupportedRequestError) as raised:
        estimate_ate(_src(), _DESIGN, methods=[Method(name="propensity-magic")])
    assert raised.value.code == "estimation.variance.registry.no_registered_available"


def test_estimate_ate_skips_ratio_metric_with_warning_but_keeps_others():
    """A mixed-type metric list must not let one unsupported metric (ratio,
    under IPTW) abort every other metric's estimate - it is skipped with a
    warning naming it, and the mean metric's estimate still comes back."""
    rng = np.random.default_rng(11)
    n = 40
    x = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-0.3 * x))).astype(float)
    revenue = 5.0 + 0.5 * x + 0.2 * d + rng.normal(scale=1.0, size=n)
    orders = np.clip(2.0 + 0.1 * x + rng.normal(scale=0.3, size=n), 0.5, None)
    tbl = pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C").tolist(),
            "revenue": revenue,
            "orders": orders,
            "x": x,
        }
    )
    src = from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(name="rpo", type="ratio", numerator="revenue", denominator="orders"),
        ],
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x",)),
        gate=IdentificationGate(overlap="trim"),
    )
    with pytest.warns(IncrementWarning) as rec:
        results = estimate_ate(src, design).results
    assert "estimation.adjust.skip_unsupported_metric" in warning_codes(rec)
    assert [r.metric for r in results] == ["revenue"]
    assert results[0].method == "iptw"


def test_mu0_zero_retains_additive_inference_without_a_ratio_point():
    """An exactly zero control mean does not erase the identified additive effect."""
    tbl = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "variant": ["T", "T", "T", "C", "C", "C"],
            "revenue": [3.0, 5.0, 4.0, 0.0, 0.0, 0.0],
            "z": [1.0] * 6,
        }
    )
    src = from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )

    def stub():
        return StubLearner([0.5] * 6)

    (result,) = estimate_ate(
        src, _DESIGN, methods=[Method(name="iptw", propensity_learner=stub)]
    ).results
    assert result.lift is None
    assert result.relative_confidence_set is not None
    assert result.relative_confidence_set.geometry == "empty"
    assert result.abs_diff == pytest.approx(4.0)
    assert result.abs_lb is not None and result.abs_lb > 0


def test_mu0_near_zero_reports_disconnected_relative_set():
    """A noisy denominator produces signed set geometry, not a finite Wald interval."""
    tbl = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "variant": ["T", "T", "T", "C", "C", "C"],
            "revenue": [3.0, 5.0, 4.0, 0.001, -0.001, 0.002],
            "z": [1.0] * 6,
        }
    )
    src = from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )

    def stub():
        return StubLearner([0.5] * 6)

    (result,) = estimate_ate(
        src, _DESIGN, methods=[Method(name="iptw", propensity_learner=stub)]
    ).results
    assert result.relative_confidence_set is not None
    assert result.relative_confidence_set.geometry == "disconnected"
    assert result.abs_diff == pytest.approx(4 - 2 / 3000)
    assert result.abs_lb is not None and result.abs_lb > 0


class _HalfPropensityStub:
    """Constant e=0.5 propensity stub, shape-adaptive for cross-fit folds."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.full(len(X), 0.5)


class _ZeroOutcomeStub:
    """m(X) == 0 everywhere: isolates the control-mean check from any
    outcome-model behavior on the extreme values below."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])


@pytest.mark.parametrize(
    "method_name,kwargs",
    [
        ("iptw", {"propensity_learner": _HalfPropensityStub}),
        (
            "aipw",
            {
                "propensity_learner": _HalfPropensityStub,
                "outcome_learner": _ZeroOutcomeStub,
                "folds": 2,
            },
        ),
        (
            "dml",
            {
                "propensity_learner": _HalfPropensityStub,
                "outcome_learner": _ZeroOutcomeStub,
                "folds": 2,
            },
        ),
    ],
)
@pytest.mark.parametrize("clustering", ["iid", "singleton", "pure_pairs"])
def test_large_scores_preserve_additive_inference_without_relative_covariance(
    method_name, kwargs, clustering
):
    from contextlib import nullcontext

    clusters = {
        "iid": list(range(8)),
        "singleton": list(range(8)),
        "pure_pairs": [0, 0, 1, 1, 2, 3, 2, 3],
    }[clustering]
    tbl = pa.table(
        {
            "user_id": [f"u{i}" for i in range(8)],
            "variant": ["T", "T", "T", "T", "C", "C", "C", "C"],
            "revenue": [1.0, 2.0, 3.0, 4.0, 1e200, -1e200, 1e200, -1e200],
            "x": [1.0] * 8,
            "cluster_id": clusters,
        }
    )
    src = from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
        cluster="cluster_id" if clustering != "iid" else None,
    )
    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
    advisory = pytest.warns(IncrementRuntimeWarning) if clustering != "iid" else nullcontext()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with advisory as rec:
            est = estimate_ate(src, design, methods=[Method(name=method_name, **kwargs)]).results[0]
    if clustering != "iid":
        assert rec is not None
        assert "estimation.engine.small_total_clusters" in warning_codes(rec)
    expected_se = {
        "iid": 5e199,
        "singleton": 5.345224838248488e199,
        "pure_pairs": 8.16496580927726e199,
    }[clustering]
    assert est.abs_diff == pytest.approx(2.5)
    assert est.abs_se == pytest.approx(expected_se, rel=1e-14)
    assert est.abs_lb == pytest.approx(-1.959963984540054 * expected_se, rel=1e-14)
    assert est.abs_ub == pytest.approx(1.959963984540054 * expected_se, rel=1e-14)
    assert est.relative_unavailable_reason == "joint_covariance_unrepresentable"
    assert est.lift is None


def test_extreme_iptw_cluster_cancellation_preserves_small_additive_variance():
    tbl = pa.table(
        {
            "user_id": list(range(8)),
            "variant": ["T"] * 4 + ["C"] * 4,
            "revenue": [1.0, 2.0, 3.0, 4.0, 1e200, -1e200, 1e200, -1e200],
            "cluster_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "x": [1.0] * 8,
        }
    )
    src = from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
        cluster="cluster_id",
    )
    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.warns(IncrementRuntimeWarning) as rec:
            est = estimate_ate(
                src, design, methods=[Method(name="iptw", propensity_learner=_HalfPropensityStub)]
            ).results[0]
    assert "estimation.engine.small_total_clusters" in warning_codes(rec)
    assert est.abs_diff == pytest.approx(2.5)
    assert est.abs_se == pytest.approx(np.sqrt(2 / 3), rel=1e-14)


@pytest.mark.parametrize(
    "method_name,kwargs,expected_se",
    [
        ("iptw", {"propensity_learner": _HalfPropensityStub}, np.sqrt(11) / 8 * 1e200),
        (
            "aipw",
            {
                "propensity_learner": _HalfPropensityStub,
                "outcome_learner": _ZeroOutcomeStub,
                "folds": 2,
            },
            np.sqrt(11) / 8 * 1e200,
        ),
        (
            "dml",
            {
                "propensity_learner": _HalfPropensityStub,
                "outcome_learner": _ZeroOutcomeStub,
                "folds": 2,
            },
            np.sqrt(46) / 16 * 1e200,
        ),
    ],
)
def test_nonzero_control_retains_large_additive_uncertainty(method_name, kwargs, expected_se):
    """Unrepresentable squared scores do not erase representable additive inference."""
    tbl = pa.table(
        {
            "user_id": [f"u{i}" for i in range(8)],
            "variant": ["T", "T", "T", "T", "C", "C", "C", "C"],
            "revenue": [1.0, 2.0, 3.0, 4.0, 1e200, -1e200, 1e200, 1.0],
            "x": [1.0] * 8,
        }
    )
    src = from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        estimate = estimate_ate(src, design, methods=[Method(name=method_name, **kwargs)]).results[
            0
        ]
    assert estimate.abs_diff == pytest.approx(-2.5e199)
    assert estimate.abs_se == pytest.approx(expected_se, rel=1e-14)
    assert estimate.relative_unavailable_reason == "joint_covariance_unrepresentable"


def test_estimate_ate_returns_keyed_failures_when_every_metric_is_refused():
    """An all-ratio metric list under IPTW returns a keyed failure bundle."""
    tbl = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["T", "T", "C", "C"],
            "revenue": [3.0, 5.0, 1.0, 2.0],
            "orders": [1.0, 2.0, 1.0, 1.0],
            "z": [1.0] * 4,
        }
    )
    src = from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[MetricSpec(name="rpo", type="ratio", numerator="revenue", denominator="orders")],
    )
    with pytest.warns(IncrementWarning) as rec:
        computation = estimate_ate(src, _DESIGN)
    assert "estimation.adjust.skip_unsupported_metric" in warning_codes(rec)
    assert computation.results == ()
    assert len(computation.failures) == 1
    failure = next(iter(computation.failures.values()))
    assert failure.hypothesis.metric == "rpo"
    # The public typed ratio refusal is documented at
    # docs/guides/observational.md:941-943.
    assert failure.code == "estimation.adjust_common.supported_ratio_metric"


def test_prepare_adjustment_requests_shares_the_ratio_and_missing_refusals():
    from increment.estimation._adjust.common import _prepare_adjustment_requests
    from increment.estimation._adjust.learners import LogisticPropensity

    tbl = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["T", "T", "C", "C"],
            "revenue": [3.0, 5.0, 1.0, 2.0],
            "orders": [1.0, 2.0, 1.0, 1.0],
            "z": [1.0] * 4,
        }
    )
    src = from_unit_summary(
        tbl,
        unit="user_id",
        group="variant",
        control="C",
        metrics=[MetricSpec(name="rpo", type="ratio", numerator="revenue", denominator="orders")],
    )
    metric = _metric(src, "rpo")
    with pytest.raises(UnsupportedRequestError) as exc_info:
        _prepare_adjustment_requests(
            src,
            metric,
            _DESIGN,
            method="AIPW",
            covariates=list(_DESIGN.adjustment.covariates),
            learner_roles=("propensity_learner", "outcome_learner"),
            propensity_learner=LogisticPropensity,
            outcome_learner=LogisticPropensity,
            prior=None,
            alpha=0.05,
            alternative="two-sided",
            null_lift=0.0,
            null_abs=None,
            value_scale="relative",
            preferred_direction=None,
        )
    assert exc_info.value.code == "estimation.adjust_common.supported_ratio_metric"


def test_prepare_adjustment_requests_excludes_zero_unit_arms():
    """A MomentSource's moments() and unit_frame() are two independent
    queries (e.g. a warehouse cube vs. a raw unit pull); when moments()
    reports a treatment arm with n == 0, `_prepare_adjustment_requests`
    must exclude it rather than build a request for an arm with zero
    units to fit on."""
    from increment.estimation._adjust.common import _prepare_adjustment_requests

    inner = _clean_confounded_src()

    class _ZeroArmSource:
        """Wraps a real source but injects a zero-`n` "T2" moment row not
        backed by any unit in unit_frame() -- the disagreement this guard
        exists for. Delegates every other MomentSource member so this
        remains a real MomentSource, not a partial stub."""

        def __init__(self, source):
            self._source = source
            self.context = source.context
            self.capabilities = source.capabilities
            self.operations = source.operations
            self.shape = source.shape
            self.breakouts = source.breakouts

        def moments(self, metric, **kwargs):
            rows = [dict(r) for r in self._source.moments(metric, **kwargs)]
            zero_row = dict(rows[0])
            zero_row["group_id"] = "T2"
            zero_row["n"] = 0
            rows.append(zero_row)
            return rows

        def unit_frame(self, metric, *, covariates=()):
            return self._source.unit_frame(metric, covariates=covariates)

        def unit_counts(self):
            return self._source.unit_counts()

        def cluster_counts(self):
            return self._source.cluster_counts()

        def compliance_dates(self):
            return self._source.compliance_dates()

        def compliance_summary(self, design, *, as_of=None, completed_windows_only=False):
            return self._source.compliance_summary(
                design, as_of=as_of, completed_windows_only=completed_windows_only
            )

        def sql(self, *, grain="total"):
            return self._source.sql(grain=grain)

        def close(self):
            return self._source.close()

    src = _ZeroArmSource(inner)
    metric = _metric(src, "revenue")
    design = _obs("refuse")
    requests, learners = _prepare_adjustment_requests(
        src,
        metric,
        design,
        method="DML",
        covariates=list(design.adjustment.covariates),
        learner_roles=("propensity_learner", "outcome_learner"),
        propensity_learner=None,
        outcome_learner=None,
        prior=None,
        alpha=0.05,
        alternative="two-sided",
        null_lift=0.0,
        null_abs=None,
        value_scale="relative",
        preferred_direction=None,
    )
    # "T2" (n == 0) is excluded; only "T" (real units) gets a request.
    assert [r.treatment_group for r in requests] == ["T"]
    assert learners == {"propensity_learner": None, "outcome_learner": None}


def test_multi_arm_produces_one_estimate_per_non_control_arm():
    """Three arms (C, T1, T2) - iptw_estimate returns one LiftEstimate per
    treatment arm, each fitted propensity predicted for every cohort unit."""
    tbl = pa.table(
        {
            "user_id": [f"u{i}" for i in range(9)],
            "variant": ["T1"] * 3 + ["T2"] * 3 + ["C"] * 3,
            "revenue": [4.0, 5.0, 3.0, 6.0, 7.0, 5.0, 1.0, 2.0, 3.0],
            "z": [1.0] * 9,
        }
    )
    src = from_unit_summary(
        tbl, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("z",)))
    results = estimate_ate(src, design).results
    assert sorted(r.group_id for r in results) == ["T1", "T2"]
    assert all(r.method == "iptw" for r in results)


class _CellMean:
    """Saturated learner for one binary covariate: the training label mean at
    each covariate level (a propensity for 0/1 labels, a conditional mean
    otherwise)."""

    def fit(self, X, d):
        x = np.asarray(X)[:, 0]
        labels = np.asarray(d, dtype=float)
        self._means = {level: float(labels[x == level].mean()) for level in np.unique(x)}

    def predict(self, X):
        return np.array([self._means[level] for level in np.asarray(X)[:, 0]])


# Per arm: (units, outcome) at x=0 and x=1. Propensities are (.4, .4, .2) at x=0
# and (.1, .1, .8) at x=1 over equal x halves, so population means are (4, 7, 8).
# T1 vs C alone mixes x differently: control mean 2.8 and relative lift 9/14,
# not 0.75.
_THREE_ARM_CELLS = {
    "C": ((40, 2.0), (10, 6.0)),
    "T1": ((40, 3.0), (10, 11.0)),
    "T2": ((20, 9.0), (80, 7.0)),
}


def _three_arm_table(folds: int = 5, *, t2_shift: float = 0.0) -> pa.Table:
    """The three-arm cohort above. x levels are spread evenly across the
    deterministic cross-fitting folds, so every training fold keeps each
    (arm, x) cell's population share and saturated learners recover the
    population nuisances exactly."""
    from increment.estimation._adjust.overlap import _fold_ids

    arms = np.array(
        [arm for arm, cells in _THREE_ARM_CELLS.items() for _ in range(sum(n for n, _ in cells))]
    )
    ids = np.array([f"u{i:03d}" for i in range(arms.size)])
    codes = np.searchsorted(np.array(["C", "T1", "T2"]), arms)
    fold = _fold_ids(ids, codes, folds)
    x = np.zeros(arms.size)
    y = np.empty(arms.size)
    for code, (arm, ((_, y0), (n1, y1))) in enumerate(_THREE_ARM_CELLS.items()):
        for j in range(folds):
            x[np.flatnonzero((codes == code) & (fold == j))[: n1 // folds]] = 1.0
        in_arm = codes == code
        y[in_arm] = np.where(x[in_arm] == 1.0, y1, y0) + (t2_shift if arm == "T2" else 0.0)
    return pa.table({"user_id": ids, "variant": arms, "revenue": y, "x": x})


_THREE_ARM_DESIGN = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))


def _three_arm_rows(method: str, table: pa.Table | None = None) -> dict[str, Any]:
    src = from_unit_summary(
        _three_arm_table() if table is None else table,
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    config = (
        Method(name=method, propensity_learner=_CellMean)
        if method == "iptw"
        else Method(name=method, propensity_learner=_CellMean, outcome_learner=_CellMean, folds=5)
    )
    results = estimate_ate(src, _THREE_ARM_DESIGN, methods=[config]).results
    return {row.group_id: row for row in results}


def _reference(row) -> tuple[float, float, float, float, float]:
    assert row.relative_confidence_set is not None
    ref = row.relative_confidence_set.reference
    return ref.a, ref.c, ref.var_a, ref.var_c, ref.cov_ac


@pytest.mark.parametrize("method", ["iptw", "aipw"])
def test_multi_arm_ate_rows_share_the_population_control_mean(method):
    """Every arm mean averages over the whole cohort: ATEs (3, 4) against
    one control mean 4, relative lifts (0.75, 1.0) -- not the pair-population
    9/14 a treatment/control-only reanalysis gives the first arm."""
    rows = _three_arm_rows(method)
    for group, (tau, lift) in {"T1": (3.0, 0.75), "T2": (4.0, 1.0)}.items():
        row = rows[group]
        assert row.estimand == "ate" and row.population is None
        assert row.require_lift().value == pytest.approx(lift, rel=1e-12)
        assert row.abs_diff == pytest.approx(tau, rel=1e-12)
        assert _reference(row)[1] == pytest.approx(4.0, rel=1e-12)


def test_multi_arm_iptw_fixed_propensity_covariance_shares_the_control_influence():
    """Fixed-propensity Hajek arm influences (N w_a / W_a)(Y - mu_a), aligned
    by unit: var(tau_1) = .625, var(mu_0) = .125 and cov = -.125 enumerated
    over the cohort (the shared control influence enters with a minus sign)."""
    _, _, var_a, var_c, cov = _reference(_three_arm_rows("iptw")["T1"])
    assert (var_a, var_c, cov) == pytest.approx((0.625, 0.125, -0.125), rel=1e-12)


def test_multi_arm_aipw_covariance_keeps_every_cohort_units_population_term():
    """With exact outcome models the arm influences are m_a(X) - mu_a on every
    unit, other arms included: the enumerated joint covariances are
    (.02, .02, .02) for T1 and (.045, .02, -.03) for T2. Dropping third-arm
    units or padding them with zeros changes both."""
    rows = _three_arm_rows("aipw")
    assert _reference(rows["T1"])[2:] == pytest.approx((0.02, 0.02, 0.02), rel=1e-12)
    assert _reference(rows["T2"])[2:] == pytest.approx((0.045, 0.02, -0.03), rel=1e-12)


def test_multi_arm_dml_keeps_each_pair_slope_over_the_common_control_mean():
    """Pair PLR slopes are propensity-weighted, not the population ATEs: 1.8
    for T1 (weight p_1 p_0 / (p_1 + p_0)) and 4.6 for T2. Relative rows divide
    them by the common augmented control mean 4 (0.45, 1.15), never by a
    pair-population control mean. The slope score is zero off its pair but is
    normalized over all 200 units: (var_theta, var_mu0, cov) = (.0256, .02,
    .0128) for T1."""
    rows = _three_arm_rows("dml")
    t1, t2 = rows["T1"], rows["T2"]
    assert t1.estimand == t2.estimand == "plr_slope"
    assert t1.abs_diff == pytest.approx(1.8, rel=1e-12)
    assert t1.require_lift().value == pytest.approx(0.45, rel=1e-12)
    assert _reference(t1)[1:] == pytest.approx((4.0, 0.0256, 0.02, 0.0128), rel=1e-12)
    assert t2.abs_diff == pytest.approx(4.6, rel=1e-12)
    assert t2.require_lift().value == pytest.approx(1.15, rel=1e-12)


@pytest.mark.parametrize("method", ["iptw", "aipw", "dml"])
def test_other_treatment_outcomes_do_not_move_a_comparison(method):
    """Another treatment's outcomes train only that arm's own models: shifting
    every T2 outcome changes T2's row and leaves T1's comparison untouched."""
    base = _three_arm_rows(method)
    shifted = _three_arm_rows(method, _three_arm_table(t2_shift=5.0))
    assert shifted["T1"].require_lift().value == pytest.approx(
        base["T1"].require_lift().value, rel=1e-12
    )
    assert _reference(shifted["T1"]) == pytest.approx(_reference(base["T1"]), rel=1e-12)
    assert shifted["T2"].abs_diff != pytest.approx(base["T2"].abs_diff)


class _SaturatingWhereX:
    """Every treatment-versus-control fit predicts 0.985 where x == 1, else 0.5."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.where(np.asarray(X)[:, 0] == 1.0, 0.985, 0.5)


def _overlap_src():
    """Ten units per arm; only two T2 units have x == 1. There each pair's
    conditional 0.985 sits inside [0.01, 0.99], but the coupled control
    propensity is 1 / (1 + 2 * 0.985 / 0.015) ~ 0.0076."""
    arms = ["C"] * 10 + ["T1"] * 10 + ["T2"] * 10
    return from_unit_summary(
        pa.table(
            {
                "user_id": [f"u{i}" for i in range(30)],
                "variant": arms,
                "revenue": [float(i % 7) + 1.0 for i in range(30)],
                "x": [1.0 if i in (20, 21) else 0.0 for i in range(30)],
            }
        ),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )


def test_overlap_gate_refuses_on_any_marginal_arm_propensity():
    """Support is judged on every marginal arm propensity over the whole
    cohort, so the T1 comparison refuses too although its own rows and pair
    conditionals are all well inside the band."""
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            _overlap_src(),
            _THREE_ARM_DESIGN,
            methods=[Method(name="iptw", propensity_learner=_SaturatingWhereX)],
        )
    assert exc_info.value.code == "adjust.identification.overlap_gate_refused"
    assert exc_info.value.context["outside_by_arm"] == (
        ("C", 0, 10),
        ("T1", 0, 10),
        ("T2", 2, 10),
    )


def test_overlap_trim_retains_one_population_for_every_comparison():
    design = _THREE_ARM_DESIGN.model_copy(update={"gate": IdentificationGate(overlap="trim")})
    results = estimate_ate(
        _overlap_src(), design, methods=[Method(name="iptw", propensity_learner=_SaturatingWhereX)]
    ).results
    assert {row.group_id for row in results} == {"T1", "T2"}
    for row in results:
        assert row.estimand == "overlap_subpopulation_ate"
        assert row.population == (
            "overlap with every marginal arm propensity >= 0.01 (28 of 30 units)"
        )


def _coupled_logistic_src():
    """36 units, x in (-1, 0, 1) balanced within every arm, so each default
    treatment-versus-control logistic fit is exactly flat at q = 1/2."""
    i = np.arange(36)
    x = np.tile([-1.0, 0.0, 1.0], 12)
    arm = (i // 3) % 3
    y = 3 + 0.7 * x + (arm == 1) * (1 + x) + (arm == 2) * (2 - 0.5 * x) + 0.2 * np.sin(i)
    return from_unit_summary(
        pa.table(
            {
                "user_id": [f"u{k}" for k in i],
                "variant": np.array(["C", "T1", "T2"])[arm],
                "revenue": y,
                "x": x,
            }
        ),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )


def test_multi_arm_logistic_iptw_covariance_includes_every_fitted_propensity():
    """Independent oracle: refit both unpenalized conditional logits under
    centered case-weight perturbations, couple, and differentiate the Hajek
    means numerically (max error 4.5e-9 against the stacked correction). Its
    (var_tau_1, var_mu_0, cov) = (.022335151932810, .010344814226642,
    .011421499344982); the fixed-propensity influence gives (.19136, .02790,
    -.02790) instead."""
    (row,) = (
        r
        for r in estimate_ate(_coupled_logistic_src(), _THREE_ARM_DESIGN).results
        if r.group_id == "T1"
    )
    a, c, var_a, var_c, cov = _reference(row)
    # At q = 1/2 every marginal propensity is 1/3: Hajek means are arm means.
    assert (a, c) == pytest.approx((0.9491029570658758, 3.0249059770925597), rel=1e-12)
    assert (var_a, var_c, cov) == pytest.approx(
        (0.022335151932810195, 0.010344814226642482, 0.011421499344982114), rel=1e-8
    )


# Parameter recovery: does IPTW actually recover a known effect under
# confounding, and beat the naive (unadjusted) comparison at doing so?


def _confounded_table(n: int, seed: int, effect: float = 0.2) -> pa.Table:
    """A confounded DGP: x1/x2 drive both treatment assignment and the
    outcome, with a known homogeneous true relative lift of `effect`."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.binomial(1, expit(-0.3 + 1.2 * x1 - 0.8 * x2))
    y = 1.0 + 0.9 * x1 + 0.6 * x2 + effect * d + rng.normal(scale=0.5, size=n)
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C"),
            "revenue": y,
            "x1": x1,
            "x2": x2,
        }
    )
    # E[Y(0)] = 1.0, homogeneous tau = effect => true relative lift = effect.


# True propensities leave [0.01, 0.99] for ~0.17% of units, so n>=1000
# trips the default refuse-gate near-certainly; recovery gates with "trim".
_OBS_TRIM = Observational(
    control_group="C",
    adjustment=AdjustmentSet(covariates=("x1", "x2")),
    gate=IdentificationGate(overlap="trim"),
)


def test_iptw_beats_naive_under_confounding_smoke():
    truth = 0.2
    an_obs = Analysis.from_unit_summary(
        _confounded_table(4000, seed=3),
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=_OBS_TRIM,
    )
    (iptw,) = lift_rows(lift_rows(an_obs.run()))
    an_naive = Analysis.from_unit_summary(
        _confounded_table(4000, seed=3),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    (naive,) = lift_rows(lift_rows(an_naive.run()))
    # Load-bearing comparative claim + coverage of the truth:
    iptw_lift = iptw.require_lift()
    naive_lift = naive.require_lift()
    assert abs(iptw_lift.value - truth) < abs(naive_lift.value - truth)
    assert abs(naive_lift.value - truth) > 0.10  # confounding is real
    assert iptw_lift.lb is not None and iptw_lift.ub is not None
    assert iptw_lift.lb < truth < iptw_lift.ub  # CI covers
    # 0.15, not razor-thin: sampling SD at n=4000 is ~0.04-0.05, so this
    # seed's ~0.10 error is unremarkable, a fraction of naive's ~0.6 error.
    assert abs(iptw_lift.value - truth) < 0.15


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_iptw_coverage_and_bias():
    truth, hits, points = 0.2, 0, []
    for seed in range(120):
        an = Analysis.from_unit_summary(
            _confounded_table(2000, seed=seed),
            unit="user_id",
            group="variant",
            metrics={"revenue": "mean"},
            design=_OBS_TRIM,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"(?:IPTW|DML|AIPW) covariate balance advisory",
                category=UserWarning,
            )
            (est,) = lift_rows(lift_rows(an.run()))
        est_lift = est.require_lift()
        points.append(est_lift.value)
        assert est_lift.lb is not None and est_lift.ub is not None
        hits += est_lift.lb < truth < est_lift.ub
    # Lower bound only: the IF treats propensity as known, a deliberately
    # conservative choice that over-covers (Lunceford & Davidian 2004).
    assert hits / 120 >= 0.88
    assert abs(np.mean(points) - truth) < 0.02  # near-unbiased


def test_dml_estimand_is_plr_slope_regardless_of_trim():
    """DML always identifies the partially-linear model's slope, distinct
    from IPTW/AIPW's nonparametric ATE - trimming restricts the input
    population but does not change that functional-form label."""

    src = from_unit_summary(
        _confounded_table(500, seed=3),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    (untrimmed,) = estimate_ate(
        src,
        _obs("refuse"),
        methods=[
            Method(
                name="dml",
                propensity_learner=LogisticPropensity,
                outcome_learner=RidgeOutcome,
                folds=2,
            )
        ],
    ).results
    assert untrimmed.population is None
    assert untrimmed.estimand == "plr_slope"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (trimmed,) = estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[
                Method(
                    name="dml",
                    propensity_learner=LogisticPropensity,
                    outcome_learner=RidgeOutcome,
                    folds=2,
                )
            ],
        ).results
    assert trimmed.estimand == "plr_slope"


def test_iptw_and_aipw_estimand_is_ate_without_trim():
    src = from_unit_summary(
        _confounded_table(500, seed=3),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (i_est,) = estimate_ate(
            src,
            _obs("refuse"),
            methods=[Method(name="iptw", propensity_learner=LogisticPropensity)],
        ).results
        (a_est,) = estimate_ate(
            src,
            _obs("refuse"),
            methods=[
                Method(
                    name="aipw",
                    propensity_learner=LogisticPropensity,
                    outcome_learner=RidgeOutcome,
                    folds=2,
                )
            ],
        ).results
    assert i_est.population is None and i_est.estimand == "ate"
    assert a_est.population is None and a_est.estimand == "ate"


def test_iptw_and_aipw_estimand_is_overlap_subpopulation_ate_when_trimmed():
    src = from_unit_summary(
        _confounded_table(2000, seed=3),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (i_est,) = estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[Method(name="iptw", propensity_learner=LogisticPropensity)],
        ).results
        (a_est,) = estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[
                Method(
                    name="aipw",
                    propensity_learner=LogisticPropensity,
                    outcome_learner=RidgeOutcome,
                    folds=2,
                )
            ],
        ).results
    assert i_est.population is not None and i_est.estimand == "overlap_subpopulation_ate"
    assert a_est.population is not None and a_est.estimand == "overlap_subpopulation_ate"


# Missing covariate values: a NaN covariate's SMD is NaN, so the gate never
# fires on it - the check must run BEFORE any learner fit and before either gate.


class NaNTolerantLearner:
    """A LightGBM-style stub: happily returns finite propensities no
    matter how many NaNs the covariate matrix carries."""

    def fit(self, X, d):
        self._p = float(np.asarray(d, dtype=float).mean())

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self._p)


def _missing_x1_table(n: int, seed: int, n_missing: int) -> pa.Table:
    """The confounded DGP with the first *n_missing* x1 values nulled."""
    table = _confounded_table(n, seed)
    x1 = table["x1"].to_pylist()
    x1[:n_missing] = [None] * n_missing
    return table.set_column(table.schema.get_field_index("x1"), "x1", pa.array(x1))


def _obs(missing: Any, **gate_kwargs) -> Observational:
    return Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x1", "x2"), missing=missing),
        gate=IdentificationGate(**gate_kwargs),
    )


def test_missing_covariate_refuses_before_gates_even_with_nan_tolerant_learner():
    """NaN + a NaN-tolerant learner + an explicit max_smd
    previously shipped a confident wrong number past the gate. The
    missingness check must fire before any learner fit."""

    class _RefusesToFit(NaNTolerantLearner):
        def fit(self, X, d):
            raise AssertionError("learner fit ran before the missingness refusal")

    src = from_unit_summary(
        _missing_x1_table(600, seed=3, n_missing=60),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    design = _obs("refuse", max_smd=0.05)
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, design, methods=[Method(name="iptw", propensity_learner=_RefusesToFit)])
    assert exc_info.value.code == "adjust.identification.missing_covariates"


def test_missing_covariate_refuses_under_default_learner_too():
    src = from_unit_summary(
        _missing_x1_table(600, seed=3, n_missing=60),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, _obs("refuse"))
    assert exc_info.value.code == "adjust.identification.missing_covariates"


def test_impute_indicator_keeps_every_unit_and_stamps_note():
    src = from_unit_summary(
        _missing_x1_table(2000, seed=3, n_missing=200),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    design = _obs("impute-indicator", overlap="trim")
    (est,) = estimate_ate(src, design).results
    assert est.note is not None
    # Still a working adjustment: closer to truth 0.2 than the raw
    # confounded contrast (~0.8 on this DGP).
    assert abs(est.require_lift().value - 0.2) < 0.15


def test_impute_indicator_flag_is_gated_like_any_covariate():
    """The missingness indicator ENTERS the adjustment set: with
    missingness concentrated in one arm and a learner that cannot balance
    anything, the indicator's own SMD must trip an explicit max_smd gate,
    named as x1__missing."""
    table = _confounded_table(300, seed=5)
    variant = table["variant"].to_pylist()
    x1 = table["x1"].to_pylist()
    treated_idx = [i for i, v in enumerate(variant) if v == "T"][:80]
    for i in treated_idx:  # missingness almost entirely in the T arm
        x1[i] = None
    table = table.set_column(table.schema.get_field_index("x1"), "x1", pa.array(x1))
    src = from_unit_summary(
        table, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    design = _obs("impute-indicator", max_smd=0.1)
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src, design, methods=[Method(name="iptw", propensity_learner=NaNTolerantLearner)]
        )
    assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"


def test_complete_case_relabels_population_like_trim():
    src = from_unit_summary(
        _missing_x1_table(2000, seed=3, n_missing=200),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    design = _obs("complete-case", overlap="trim")
    (est,) = estimate_ate(src, design).results
    assert est.population is not None
    # The 200 units with a missing x1 are dropped: the complete-case estimate is
    # the same design's estimate over exactly the rows that carry x1.
    complete_rows = from_unit_summary(
        _confounded_table(2000, seed=3).slice(200),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    (oracle,) = estimate_ate(complete_rows, _obs("refuse", overlap="trim")).results
    assert est.require_lift().value == pytest.approx(oracle.require_lift().value, rel=1e-12)


def test_pattern_fits_per_missingness_pattern_and_stamps_note():
    src = from_unit_summary(
        _missing_x1_table(2000, seed=3, n_missing=200),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    design = _obs("pattern", overlap="trim")
    (est,) = estimate_ate(src, design).results
    assert est.note is not None
    assert abs(est.require_lift().value - 0.2) < 0.15


def test_pattern_refuses_thin_pattern_naming_alternatives():
    src = from_unit_summary(
        _missing_x1_table(600, seed=3, n_missing=10),  # 10 < 30-unit floor
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, _obs("pattern", overlap="trim"))
    assert exc_info.value.code == "adjust.identification.pattern_below_floor"


def test_pattern_refuses_scattered_missingness_over_the_cap():
    rng = np.random.default_rng(9)
    n = 400
    cols = {f"c{i}": rng.normal(size=n) for i in range(6)}
    for i in range(6):  # scattered missingness -> combinatorial patterns
        idx = rng.choice(n, 80, replace=False)
        col = cols[f"c{i}"].astype(object)
        col[idx] = None
        cols[f"c{i}"] = col
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": ["T" if i % 2 else "C" for i in range(n)],
            "revenue": rng.normal(loc=4.0, size=n),
            **{k: pa.array(list(v), type=pa.float64()) for k, v in cols.items()},
        }
    )
    src = from_unit_summary(
        table, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=tuple(cols), missing="pattern"),
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, design)
    assert exc_info.value.code == "adjust.identification.pattern_cap_exceeded"


def test_dml_missing_covariate_refuses_and_impute_indicator_unblocks():

    src = from_unit_summary(
        _missing_x1_table(2000, seed=3, n_missing=200),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, _obs("refuse", overlap="trim"), methods=[Method(name="dml")])
    assert exc_info.value.code == "adjust.identification.missing_covariates"
    (est,) = estimate_ate(
        src, _obs("impute-indicator", overlap="trim"), methods=[Method(name="dml")]
    ).results
    assert est.note is not None
    assert abs(est.require_lift().value - 0.2) < 0.15


def test_dml_pattern_refused_naming_the_supported_paths():

    src = from_unit_summary(
        _missing_x1_table(600, seed=3, n_missing=60),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, _obs("pattern", overlap="trim"), methods=[Method(name="dml")])
    assert exc_info.value.code == "adjust.identification.pattern_requires_iptw"


# Non-finite/out-of-range nuisances must refuse on the DEFAULT
# (non-allow) missing policy too, not just under missing="allow" - the
# gates below compare False against NaN and would otherwise silently ship
# an all-NaN estimate.


class _NaNPropensityLearner:
    """Silently emits one NaN propensity - the measured default-learner
    failure mode, independent of any missing-covariate policy."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        p = np.full(np.asarray(X).shape[0], 0.5)
        p[0] = np.nan
        return p


class _NaNOutcomeLearner:
    def fit(self, X, d):
        pass

    def predict(self, X):
        p = np.full(np.asarray(X).shape[0], 0.0)
        p[0] = np.nan
        return p


class _OutOfRangePropensityLearner:
    """A broken propensity model (leaves [0, 1] entirely) rather than a
    positivity violation - must not be silently trimmed away as if it
    were a subpopulation."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        p = np.full(np.asarray(X).shape[0], 0.5)
        p[0] = -0.3
        return p


class _AboveRangePropensityLearner:
    """Same defect as _OutOfRangePropensityLearner but on the upper side:
    a linear-probability model that predicts e > 1.0, which must trip the
    same refusal (and surface a positive max in the message)."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        p = np.full(np.asarray(X).shape[0], 0.5)
        p[0] = 1.3
        return p


def _clean_confounded_src(n=200, seed=3):
    return from_unit_summary(
        _confounded_table(n, seed),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )


def test_iptw_default_path_refuses_nonfinite_propensity():
    src = _clean_confounded_src()
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[Method(name="iptw", propensity_learner=_NaNPropensityLearner)],
        )
    assert exc_info.value.code == "adjust.identification.nonfinite_nuisance"


def test_iptw_default_path_refuses_out_of_range_propensity():
    src = _clean_confounded_src()
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[Method(name="iptw", propensity_learner=_OutOfRangePropensityLearner)],
        )
    assert exc_info.value.code == "adjust.identification.propensity_out_of_range"


def test_iptw_default_path_refuses_above_range_propensity():
    src = _clean_confounded_src()
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[Method(name="iptw", propensity_learner=_AboveRangePropensityLearner)],
        )
    assert exc_info.value.code == "adjust.identification.propensity_out_of_range"


def test_dml_default_path_refuses_nonfinite_propensity_and_outcome():

    src = _clean_confounded_src()
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[
                Method(
                    name="dml",
                    propensity_learner=_NaNPropensityLearner,
                    outcome_learner=RidgeOutcome,
                    folds=2,
                )
            ],
        )
    assert exc_info.value.code == "adjust.identification.nonfinite_nuisance"
    assert exc_info.value.context["what"] == "propensity"
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[
                Method(
                    name="dml",
                    propensity_learner=LogisticPropensity,
                    outcome_learner=_NaNOutcomeLearner,
                    folds=2,
                )
            ],
        )
    assert exc_info.value.code == "adjust.identification.nonfinite_nuisance"
    assert exc_info.value.context["what"] == "outcome"


def test_dml_default_path_refuses_out_of_range_propensity():

    src = _clean_confounded_src()
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[
                Method(
                    name="dml",
                    propensity_learner=_OutOfRangePropensityLearner,
                    outcome_learner=RidgeOutcome,
                    folds=2,
                )
            ],
        )
    assert exc_info.value.code == "adjust.identification.propensity_out_of_range"


def test_aipw_default_path_refuses_nonfinite_nuisances():
    src = _clean_confounded_src()
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[
                Method(
                    name="aipw",
                    propensity_learner=_NaNPropensityLearner,
                    outcome_learner=RidgeOutcome,
                    folds=2,
                )
            ],
        )
    assert exc_info.value.code == "adjust.identification.nonfinite_nuisance"
    assert exc_info.value.context["what"] == "propensity"
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[
                Method(
                    name="aipw",
                    propensity_learner=LogisticPropensity,
                    outcome_learner=_NaNOutcomeLearner,
                    folds=2,
                )
            ],
        )
    assert exc_info.value.code == "adjust.identification.nonfinite_nuisance"
    assert exc_info.value.context["what"] == "treatment-arm outcome"


def test_aipw_default_path_refuses_out_of_range_propensity():
    src = _clean_confounded_src()
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            _obs("refuse", overlap="trim"),
            methods=[
                Method(
                    name="aipw",
                    propensity_learner=_OutOfRangePropensityLearner,
                    outcome_learner=RidgeOutcome,
                    folds=2,
                )
            ],
        )
    assert exc_info.value.code == "adjust.identification.propensity_out_of_range"


# Missing covariate values (missing="allow"): NaN routed raw to an explicit
# NaN-native learner; three refusal layers guard the seam (config/probe/post-fit finiteness).


class MIABoost:
    """Test-only missingness-aware gradient boosting: every split learns a
    routing direction for NaN (LightGBM/HistGB-style MIA), xgboost-style
    leaf values (lambda=1). depth>=2 can represent pattern x covariate
    structure; depth=1 is the additive negative control."""

    def __init__(self, *, depth=2, n_trees=100, lr=0.1, loss="logistic", n_bins=16):
        self.depth = depth
        self.n_trees = n_trees
        self.lr = lr
        self.loss = loss
        self.n_bins = n_bins

    def _codes(self, X, nan):
        return np.column_stack(
            [
                np.searchsorted(self._edges[j], np.where(nan[:, j], 0.0, X[:, j]), side="right")
                for j in range(X.shape[1])
            ]
        )

    def fit(self, X, d):
        X = np.asarray(X, dtype=float)
        y = np.asarray(d, dtype=float)
        n, p = X.shape
        nan = np.isnan(X)
        self._edges = []
        for j in range(p):
            obs = X[~nan[:, j], j]
            qs = np.quantile(obs, np.linspace(0, 1, self.n_bins + 1)[1:-1]) if obs.size else []
            self._edges.append(np.unique(qs))
        codes = self._codes(X, nan)
        if self.loss == "logistic":
            base = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            self._f0 = float(np.log(base / (1.0 - base)))
        else:
            self._f0 = float(y.mean())
        f = np.full(n, self._f0)
        self._trees = []
        rows = np.arange(n)
        for _ in range(self.n_trees):
            if self.loss == "logistic":
                prob = expit(f)
                g, h = prob - y, prob * (1.0 - prob)
            else:
                g, h = f - y, np.ones(n)
            tree = self._build(codes, nan, g, h, rows, self.depth)
            self._trees.append(tree)
            f += self.lr * self._apply(tree, codes, nan)

    def _build(self, codes, nan, g, h, rows, depth):
        big_g, big_h = g[rows].sum(), h[rows].sum()
        leaf = float(-big_g / (big_h + 1.0))
        if depth == 0 or rows.size < 8:
            return leaf
        parent = big_g * big_g / (big_h + 1.0)
        best = None  # (gain, feature, code threshold, nan-goes-left)
        for j in range(codes.shape[1]):
            nan_j = nan[rows, j]
            obs_rows = rows[~nan_j]
            if obs_rows.size == 0:
                continue
            g_nan, h_nan = g[rows[nan_j]].sum(), h[rows[nan_j]].sum()
            nb = len(self._edges[j]) + 1
            sums_g = np.bincount(codes[obs_rows, j], weights=g[obs_rows], minlength=nb)
            sums_h = np.bincount(codes[obs_rows, j], weights=h[obs_rows], minlength=nb)
            gl, hl = np.cumsum(sums_g), np.cumsum(sums_h)
            g_obs, h_obs = gl[-1], hl[-1]
            for nan_left in (True, False):
                glt = gl + (g_nan if nan_left else 0.0)
                hlt = hl + (h_nan if nan_left else 0.0)
                grt = (g_obs - gl) + (0.0 if nan_left else g_nan)
                hrt = (h_obs - hl) + (0.0 if nan_left else h_nan)
                gains = glt**2 / (hlt + 1.0) + grt**2 / (hrt + 1.0) - parent
                b = int(np.argmax(gains))
                if best is None or gains[b] > best[0]:
                    best = (float(gains[b]), j, b, nan_left)
        if best is None or best[0] <= 1e-12:
            return leaf
        _, j, b, nan_left = best
        go_left = np.where(nan[rows, j], nan_left, codes[rows, j] <= b)
        if not go_left.any() or go_left.all():
            return leaf
        return (
            j,
            b,
            nan_left,
            self._build(codes, nan, g, h, rows[go_left], depth - 1),
            self._build(codes, nan, g, h, rows[~go_left], depth - 1),
        )

    def _apply(self, tree, codes, nan):
        if not isinstance(tree, tuple):
            return np.full(codes.shape[0], tree)
        j, b, nan_left, left, right = tree
        go_left = np.where(nan[:, j], nan_left, codes[:, j] <= b)
        out = np.empty(codes.shape[0])
        out[go_left] = self._apply(left, codes[go_left], nan[go_left])
        out[~go_left] = self._apply(right, codes[~go_left], nan[~go_left])
        return out

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        nan = np.isnan(X)
        codes = self._codes(X, nan)
        f = np.full(X.shape[0], self._f0)
        for tree in self._trees:
            f += self.lr * self._apply(tree, codes, nan)
        return expit(f) if self.loss == "logistic" else f


# Pinned MAR DGP: Z observed, X missing w.p. expit(Z+shift), pattern-flipped
# propensity/outcome so MPA identification holds. tau=0.5, mu0=1.6248 at shift -0.8.
_ALLOW_TRUTH = 0.5 / 1.6248


def _mar_pattern_table(n, seed, shift=-0.8):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    miss = rng.random(n) < expit(z + shift)
    e = np.where(miss, expit(-1.2 * z), expit(1.2 * x + 0.8 * z))
    d = rng.binomial(1, e)
    y = 2.0 + np.where(miss, -z, x + z) + (0.5 + 0.3 * z) * d + rng.normal(size=n)
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "variant": np.where(d == 1, "T", "C"),
            "revenue": y,
            "x": np.where(miss, np.nan, x),
            "z": z,
        }
    )


def _mar_src(n, seed, shift=-0.8):
    return from_unit_summary(
        _mar_pattern_table(n, seed, shift),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )


def _mar_design(missing, **gate_kwargs):
    return Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x", "z"), missing=missing),
        gate=IdentificationGate(**gate_kwargs),
    )


_SMD_ROW = re.compile(r"(?:: |, )([\w*]+) \(SMD=(-?[\d.]+)\)")


def _smd_rows(messages):
    """Parse `name (SMD=value)` rows out of gate advisory/refusal text."""
    rows = {}
    for msg in messages:
        rows.update({name: float(v) for name, v in _SMD_ROW.findall(msg)})
    return rows


def _run_allow_aipw(n, seed, *, depth, pt=300, ot=150, lr=0.1, shift=-0.8, max_smd=None):
    """One aipw run under missing='allow' with the MIA stub; returns the
    estimate and any captured balance-advisory messages."""
    src = _mar_src(n, seed, shift)
    gate = {"overlap": "trim"} | ({"max_smd": max_smd} if max_smd is not None else {})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (est,) = estimate_ate(
            src,
            _mar_design("allow", **gate),
            methods=[
                Method(
                    name="aipw",
                    propensity_learner=lambda: MIABoost(depth=depth, n_trees=pt, lr=lr),
                    outcome_learner=lambda: MIABoost(
                        depth=depth, n_trees=ot, lr=lr, loss="squared"
                    ),
                    folds=3,
                )
            ],
        ).results
    return est, [str(w.message) for w in caught if "balance advisory" in str(w.message)]


class _ProbeOnlyFinite:
    """Passes the small synthetic NaN probe but emits one NaN prediction on
    the real data - only the post-fit finiteness guard can catch it."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        n = np.asarray(X).shape[0]
        out = np.full(n, 0.5)
        if n > 64:
            out[0] = np.nan
        return out


class _RaisesOnRealFit:
    """Passes the 64-row probe, raises on any larger (real) fit."""

    def fit(self, X, d):
        if np.asarray(X).shape[0] > 64:
            raise RuntimeError("cannot fit this data")

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], 0.5)


class _RaisesOnAllNaNColumn:
    """A NaN-native learner that raises when a training column is entirely
    NaN - the fold x arm subset case the exception wrapping exists for."""

    def fit(self, X, d):
        if np.isnan(np.asarray(X, dtype=float)).all(axis=0).any():
            raise ValueError("training column is entirely NaN")

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], 0.5)


class _RaisesOnNaN:
    """A learner that raises on any NaN in training data."""

    def fit(self, X, d):
        if np.isnan(np.asarray(X, dtype=float)).any():
            raise ValueError("NaN in training data")

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], 0.5)


class _PresetPropensity:
    """Preset per-row propensities on the real data, 0.5 on the probe."""

    def __init__(self, e):
        self._e = np.asarray(e, dtype=float)

    def fit(self, X, d):
        pass

    def predict(self, X):
        n = np.asarray(X).shape[0]
        return self._e if n == self._e.shape[0] else np.full(n, 0.5)


def test_allow_refuses_defaulted_learners_unconditionally():
    """Layer 1: 'allow' + any defaulted X-consuming learner is a config
    error even on completely clean (zero-NaN) data - the defaults
    silently emit non-finite predictions under NaN."""

    src = from_unit_summary(
        _confounded_table(60, seed=1),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, _obs("allow"))
    assert exc_info.value.code == "adjust.identification.allow_requires_learner"
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, _obs("allow"), methods=[Method(name="dml")])
    assert exc_info.value.code == "adjust.identification.allow_requires_learner"
    # Supplying ONE factory is not enough: the outcome models see X too.
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src, _obs("allow"), methods=[Method(name="aipw", propensity_learner=NaNTolerantLearner)]
        )
    assert exc_info.value.code == "adjust.identification.allow_requires_learner"


_PROBE_CODES = frozenset(
    {
        "adjust.identification.allow_probe_raised",
        "adjust.identification.allow_probe_nonfinite",
    }
)


def test_allow_probe_rejects_learners_that_cannot_take_nan():
    """Layer 2: the probe refuses a learner that raises on NaN and one that
    silently returns non-finite predictions (the closed-form RidgeOutcome).
    The default LogisticPropensity does one or the other depending on whether
    the NumPy build warns on NaN arithmetic, so only its refusal is pinned."""
    src = from_unit_summary(
        _missing_x1_table(600, seed=3, n_missing=60),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src, _obs("allow"), methods=[Method(name="iptw", propensity_learner=_RaisesOnNaN)]
        )
    assert exc_info.value.code == "adjust.identification.allow_probe_raised"
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src, _obs("allow"), methods=[Method(name="iptw", propensity_learner=LogisticPropensity)]
        )
    assert exc_info.value.code in _PROBE_CODES
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            _obs("allow"),
            methods=[
                Method(
                    name="aipw",
                    propensity_learner=NaNTolerantLearner,
                    outcome_learner=RidgeOutcome,
                    folds=2,
                )
            ],
        )
    assert exc_info.value.code == "adjust.identification.allow_probe_nonfinite"


def test_allow_post_fit_guard_refuses_real_data_nan():
    """Layer 3: a learner can pass the probe and still emit NaN on a real
    region - the post-fit guard names the learner and the count."""
    src = from_unit_summary(
        _missing_x1_table(200, seed=3, n_missing=20),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src, _obs("allow"), methods=[Method(name="iptw", propensity_learner=_ProbeOnlyFinite)]
        )
    assert exc_info.value.code == "adjust.identification.nonfinite_nuisance"


def test_allow_refuses_infinite_covariates_by_name():
    """Only NaN is routable missingness under 'allow': a NaN-native
    learner's split routing has no +/-inf semantics."""
    table = _confounded_table(200, seed=5)
    x1 = table["x1"].to_pylist()
    x1[:3] = [float("inf")] * 3
    table = table.set_column(table.schema.get_field_index("x1"), "x1", pa.array(x1))
    src = from_unit_summary(
        table, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src, _obs("allow"), methods=[Method(name="iptw", propensity_learner=NaNTolerantLearner)]
        )
    assert exc_info.value.code == "adjust.identification.allow_nonfinite_covariates"


def test_allow_all_nan_column_refuses_while_building_gate_matrix():
    table = _confounded_table(120, seed=5)
    table = table.set_column(
        table.schema.get_field_index("x1"), "x1", pa.array([None] * 120, type=pa.float64())
    )
    src = from_unit_summary(
        table, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src, _obs("allow"), methods=[Method(name="iptw", propensity_learner=NaNTolerantLearner)]
        )
    assert exc_info.value.code == "adjust.identification.covariate_fully_missing"


def test_allow_wraps_learner_raise_naming_stage():
    src = from_unit_summary(
        _missing_x1_table(200, seed=3, n_missing=20),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src, _obs("allow"), methods=[Method(name="iptw", propensity_learner=_RaisesOnRealFit)]
        )
    assert exc_info.value.code == "adjust.identification.allow_fit_predict_raised"


def test_allow_wraps_fold_arm_all_nan_column_raise():
    """A covariate observed only in one arm is ALL-NaN within every
    train & other-arm subset: the global all-NaN refusal misses it, the
    probe plants only partial NaN, so the wrap must name learner, fold,
    and arm instead of leaking a raw traceback."""
    n = 200
    rng = np.random.default_rng(2)
    variant = np.where(np.arange(n) % 2 == 0, "T", "C")
    src = from_unit_summary(
        pa.table(
            {
                "user_id": [f"u{i}" for i in range(n)],
                "variant": variant,
                "revenue": rng.normal(loc=2.0, size=n),
                "x1": np.where(variant == "T", rng.normal(size=n), np.nan),
                "x2": rng.normal(size=n),
            }
        ),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            _obs("allow"),
            methods=[
                Method(
                    name="aipw",
                    propensity_learner=NaNTolerantLearner,
                    outcome_learner=_RaisesOnAllNaNColumn,
                    folds=2,
                )
            ],
        )
    assert exc_info.value.code == "adjust.identification.allow_fit_predict_raised"


def test_allow_aipw_tolerates_fold_arm_all_nan_with_mia_stub():
    """Graceful degradation is learner-dependent: the MIA stub routes an
    all-NaN training column with a learned default direction instead of
    raising, so the same data as above yields an estimate."""
    n = 160
    rng = np.random.default_rng(2)
    variant = np.where(np.arange(n) % 2 == 0, "T", "C")
    src = from_unit_summary(
        pa.table(
            {
                "user_id": [f"u{i}" for i in range(n)],
                "variant": variant,
                "revenue": rng.normal(loc=2.0, size=n),
                "x1": np.where(variant == "T", rng.normal(size=n), np.nan),
                "x2": rng.normal(size=n),
            }
        ),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (est,) = estimate_ate(
            src,
            _obs("allow"),
            methods=[
                Method(
                    name="aipw",
                    propensity_learner=NaNTolerantLearner,
                    outcome_learner=lambda: MIABoost(depth=2, n_trees=20, loss="squared"),
                    folds=2,
                )
            ],
        ).results
    assert np.isfinite(est.require_lift().value)
    assert est.note is not None


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
def test_allow_zero_nan_runs_identical_to_refuse_path():
    """With no missing covariates, allowing missingness changes no inference."""
    table = _confounded_table(300, seed=11)
    src = from_unit_summary(
        table, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}
    )
    (i_allow,) = estimate_ate(
        src,
        _obs("allow", overlap="trim"),
        methods=[Method(name="iptw", propensity_learner=LogisticPropensity)],
    ).results
    (i_refuse,) = estimate_ate(
        src,
        _obs("refuse", overlap="trim"),
        methods=[Method(name="iptw", propensity_learner=LogisticPropensity)],
    ).results
    assert i_allow == i_refuse
    (a_allow,) = estimate_ate(
        src,
        _obs("allow", overlap="trim"),
        methods=[
            Method(
                name="aipw",
                propensity_learner=LogisticPropensity,
                outcome_learner=RidgeOutcome,
                folds=2,
            )
        ],
    ).results
    (a_refuse,) = estimate_ate(
        src,
        _obs("refuse", overlap="trim"),
        methods=[
            Method(
                name="aipw",
                propensity_learner=LogisticPropensity,
                outcome_learner=RidgeOutcome,
                folds=2,
            )
        ],
    ).results
    assert a_allow == a_refuse


def _interaction_confounded_table():
    """Every marginal SMD is exactly 0 by construction (the x1-missing
    block's x2 imbalance is cancelled by the observed block's opposite
    imbalance); only the x1__missing*x2 interaction row (SMD 1.949, an
    unweighted-pooled-sd Austin & Stuart standardized difference) can see
    the pattern-specific confounding."""
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(40)],
            "variant": ["T"] * 20 + ["C"] * 20,
            "revenue": [3.0, 3.4] * 10 + [2.0, 2.4] * 10,
            "x1": pa.array([None] * 10 + [0.0] * 10 + [None] * 10 + [0.0] * 10, type=pa.float64()),
            "x2": [1.0] * 10 + [-1.0] * 10 + [-1.0] * 10 + [1.0] * 10,
        }
    )


def test_allow_interaction_smd_row_trips_max_smd_by_name():
    src = from_unit_summary(
        _interaction_confounded_table(),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    design = _obs("allow", max_smd=0.5)
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src, design, methods=[Method(name="iptw", propensity_learner=NaNTolerantLearner)]
        )
    assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"


def test_allow_interaction_smd_advisory_names_the_imbalance():
    src = from_unit_summary(
        _interaction_confounded_table(),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.warns(IncrementWarning) as rec:
        estimate_ate(
            src, _obs("allow"), methods=[Method(name="iptw", propensity_learner=NaNTolerantLearner)]
        )
    assert "estimation.adjust_common.covariate_balance_advisory" in warning_codes(rec)
    advisory = next(
        w.message
        for w in rec
        if getattr(w.message, "code", None) == "estimation.adjust_common.covariate_balance_advisory"
    )
    assert isinstance(advisory, IncrementWarning)
    assert "x1__missing*x2" in str(advisory.context["named"])


def _trim_order_table():
    """Two poison rows (x2 = +/-10) whose inclusion pushes the x2 SMD to
    0.684; every SMD is exactly 0 once they are trimmed away."""
    x2_block = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    return pa.table(
        {
            "user_id": [f"u{i}" for i in range(16)],
            "variant": ["T"] * 8 + ["C"] * 8,
            "revenue": [3.0, 3.4] * 4 + [2.0, 2.4] * 4,
            "x1": pa.array([None, 0.0] + [0.0] * 6 + [None, 0.0] + [0.0] * 6, type=pa.float64()),
            "x2": [0.0, 10.0, *x2_block, 0.0, -10.0, *x2_block],
        }
    )


def test_allow_trims_before_computing_gate_smds():
    """Ordering is load-bearing: the poison rows sit outside the overlap
    band, so a max_smd below their pre-trim imbalance only passes when
    the trim runs FIRST (the landed order)."""
    src = from_unit_summary(
        _trim_order_table(),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    design = _obs("allow", overlap="trim", max_smd=0.5)
    e_poison = [0.5, 0.995] + [0.5] * 6 + [0.5, 0.005] + [0.5] * 6
    (est,) = estimate_ate(
        src,
        design,
        methods=[Method(name="iptw", propensity_learner=lambda: _PresetPropensity(e_poison))],
    ).results
    assert est.population == "overlap e in [0.01, 0.99] (14 of 16 units)"
    # Sanity: with the poison rows kept in-band, the same gate refuses;
    # the pass above is trim-ordering, not a vacuous threshold.
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(
            src,
            design,
            methods=[Method(name="iptw", propensity_learner=lambda: _PresetPropensity([0.5] * 16))],
        )
    assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"


def test_allow_named_in_refuse_and_pattern_refusals():
    """The default-policy refusal and the dml/aipw pattern refusals must
    name missing='allow' with the explicit-learner requirement inline."""

    src = from_unit_summary(
        _missing_x1_table(600, seed=3, n_missing=60),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, _obs("refuse"))
    assert exc_info.value.code == "adjust.identification.missing_covariates"
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, _obs("pattern", overlap="trim"), methods=[Method(name="dml")])
    assert exc_info.value.code == "adjust.identification.pattern_requires_iptw"
    assert exc_info.value.context["method"] == "DML"
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(src, _obs("pattern", overlap="trim"), methods=[Method(name="aipw")])
    assert exc_info.value.code == "adjust.identification.pattern_requires_iptw"
    assert exc_info.value.context["method"] == "AIPW"


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_allow_capable_learner_recovers_where_additive_fails():
    """Cheap twin of the recovery/negative-control pair: depth-2 MIA lands
    nearer truth than depth-1 (additive), which ships the
    pattern-confounding bias upward AND flags the x__missing*z row.

    Marked despite being small: its claim is statistical, so it must fit
    boosted learners, and no sizing gets that under the per-test budget.
    The fast suite already pins the interaction-row MECHANICS
    deterministically with a stub learner (see the max_smd and advisory
    tests above); what is left here is the capacity contrast itself.
    """
    est2, _ = _run_allow_aipw(500, 104, depth=2, pt=50, ot=30, lr=0.2)
    est1, msgs1 = _run_allow_aipw(500, 104, depth=1, pt=50, ot=30, lr=0.2)
    assert abs(est2.require_lift().value - _ALLOW_TRUTH) < 0.4
    assert est1.require_lift().value - _ALLOW_TRUTH > 0.4
    assert abs(est2.require_lift().value - _ALLOW_TRUTH) < abs(
        est1.require_lift().value - _ALLOW_TRUTH
    )
    rows = _smd_rows(msgs1)
    assert abs(rows.get("x__missing*z", 0.0)) > 0.25


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_allow_depth2_recovery_vs_fallbacks():
    """Design-doc validation item 1 (+8): allow with an interaction-capable
    NaN-native learner recovers what impute-indicator-with-DEFAULT-learners
    cannot, with IF-based coverage near nominal; the SAME capable learners
    under impute-indicator also recover (the value of allow is the
    identification semantics + gate representation, not an accuracy edge)."""
    points, hits, chatter = [], 0, 0
    for rep in range(30):
        est, msgs = _run_allow_aipw(4000, 20260811 + rep, depth=2)
        points.append(est.require_lift().value)
        hits += est.require_lift().lb < _ALLOW_TRUTH < est.require_lift().ub
        chatter += "x__missing*z" in _smd_rows(msgs)
    assert abs(np.mean(points) - _ALLOW_TRUTH) < 0.02  # measured +0.003 (MCSE 0.007)
    assert hits >= 24  # measured 28/30
    # Interaction rows chatter near the oracle noise floor, not systematically.
    assert chatter <= 4  # measured 0/30

    def _fallback(rep, missing, **kwargs):
        src = _mar_src(4000, 20260811 + rep)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (est,) = estimate_ate(
                src,
                _mar_design(missing, overlap="trim"),
                methods=[Method(name="aipw", folds=3, **kwargs)],
            ).results
        return est.require_lift().value

    defaults = [_fallback(rep, "impute-indicator") for rep in range(5)]
    assert np.mean(defaults) - _ALLOW_TRUTH > 0.3  # measured +0.51
    capable = [
        _fallback(
            rep,
            "impute-indicator",
            propensity_learner=lambda: MIABoost(depth=2, n_trees=300),
            outcome_learner=lambda: MIABoost(depth=2, n_trees=150, loss="squared"),
        )
        for rep in range(5)
    ]
    assert abs(np.mean(capable) - _ALLOW_TRUTH) < 0.06


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_allow_additive_negative_control_flags_interaction_row():
    """Design-doc validation item 2: a depth-1 additive NaN-native learner
    passes all three refusal layers and every MARGINAL balance check while
    shipping bias > +0.3 with 0/30 coverage - only the x__missing*z
    interaction row sees it, in every rep, and gate.max_smd turns that
    advisory into a hard refusal."""
    points, hits = [], 0
    for rep in range(30):
        est, msgs = _run_allow_aipw(4000, 20260811 + rep, depth=1)
        points.append(est.require_lift().value)
        hits += est.require_lift().lb < _ALLOW_TRUTH < est.require_lift().ub
        rows = _smd_rows(msgs)
        # The interaction row flags in every rep (measured |SMD| >= 0.42)...
        assert abs(rows.get("x__missing*z", 0.0)) > 0.3
        # ...while every marginal row stays under the 0.1 advisory.
        assert set(rows) == {"x__missing*z"}
    assert np.mean(points) - _ALLOW_TRUTH > 0.35  # measured +0.44
    assert hits == 0
    with pytest.raises(IdentificationError) as exc_info:
        _run_allow_aipw(4000, 20260811, depth=1, max_smd=0.3)
    assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_allow_matches_pattern_on_structural_missingness_iptw():
    """Design-doc validation item 6: on a 2-fat-pattern DGP both
    missing='pattern' (parametric per-pattern) and missing='allow'
    (NaN-native MIA) recover the truth through iptw."""
    pattern_errs, allow_errs = [], []
    for seed in (20260811, 20260812, 20260813):
        src = _mar_src(4000, seed)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"(?:IPTW|DML|AIPW) covariate balance advisory",
                category=UserWarning,
            )
            (ep,) = estimate_ate(src, _mar_design("pattern", overlap="trim")).results
            (ea,) = estimate_ate(
                src,
                _mar_design("allow", overlap="trim"),
                methods=[
                    Method(name="iptw", propensity_learner=lambda: MIABoost(depth=2, n_trees=300))
                ],
            ).results
        pattern_errs.append(ep.require_lift().value - _ALLOW_TRUTH)
        allow_errs.append(ea.require_lift().value - _ALLOW_TRUTH)
        assert abs(pattern_errs[-1]) < 0.25 and abs(allow_errs[-1]) < 0.25
    assert abs(np.mean(pattern_errs)) < 0.1
    assert abs(np.mean(allow_errs)) < 0.1


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_allow_aipw_recovers_under_arm_skewed_rare_pattern():
    """Design-doc validation item 7a: a rare missingness pattern (~7%)
    concentrated in the control arm leaves per-fold train & treat subsets
    with few NaN rows - the MIA stub degrades gracefully and still
    recovers, with no new structural refusal."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=2_000_000)
    mu0 = 2.0 - 2.0 * float(np.mean(z * expit(z - 3.0)))
    truth = 0.5 / mu0
    errs = []
    for seed in (20260811, 20260812, 20260813, 20260814, 20260815):
        est, _ = _run_allow_aipw(4000, seed, depth=2, shift=-3.0)
        errs.append(est.require_lift().value - truth)
        assert abs(errs[-1]) < 0.1
    assert abs(float(np.mean(errs))) < 0.05  # measured +0.013


@pytest.mark.parametrize("order", [("aipw", "unadjusted"), ("unadjusted", "aipw")])
def test_unadjusted_never_takes_the_decision_role_under_an_observational_design(order):
    """estimate_ate labels `unadjusted` "confounded; not a causal estimate",
    so it must never be auto-promoted over an identified adjustment."""
    n, rng = 2000, np.random.default_rng(3)
    x = rng.normal(size=n)
    d = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.9 * x))).astype(int)
    y = 1.0 + 2.0 * x + 0.0 * d + rng.normal(scale=0.5, size=n)  # true effect 0
    src = from_unit_summary(
        pa.table(
            {
                "user_id": [f"u{i}" for i in range(n)],
                "variant": np.where(d == 1, "T", "C"),
                "revenue": y,
                "x": x,
            }
        ),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x",)),
        gate=IdentificationGate(max_smd=1.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = estimate_ate(src, design, methods=[Method(name=n_) for n_ in order])

    roles = {r.method: r.method_role for r in out.results}
    assert roles["unadjusted"] == "sensitivity"
    assert roles["aipw"] == "decision"
    assert [e.method for e in out.evidence.values()] == ["aipw"]


# Categorical adjustment columns: the raw string column must reproduce the
# careful user's modal-reference dummy oracle on every estimator, under every
# missing policy, without a new knob.

_METHODS = ("iptw", "aipw", "dml")


def _cat_source(table: pa.Table, **kwargs):
    return from_unit_summary(
        table, unit="user_id", group="variant", control="C", metrics={"revenue": "mean"}, **kwargs
    )


def _cat_design(
    covariates: tuple[str, ...],
    missing: Literal["refuse", "impute-indicator", "pattern", "complete-case", "allow"] = "refuse",
    **gate_kwargs,
):
    return Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=covariates, missing=missing),
        gate=IdentificationGate(**gate_kwargs),
    )


_RAW = ("spend", "region")
_DUMMIES = ("spend", *DUMMY_COLUMNS)


def _rows(table: pa.Table, covariates: tuple[str, ...], method: str, **design_kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = estimate_ate(
            _cat_source(table),
            _cat_design(covariates, **design_kwargs),
            methods=[Method(name=method)],
        ).results
    return [row.model_dump() for row in results]


@pytest.mark.parametrize("method", _METHODS)
def test_categorical_adjustment_matches_modal_reference_dummies(method):
    units = categorical_units(600, seed=3)
    oracle = _rows(dummy_table(units), _DUMMIES, method, overlap="trim")
    actual = _rows(raw_table(units), _RAW, method, overlap="trim")
    assert_rows_match(oracle, actual)


@pytest.mark.parametrize("method", _METHODS)
def test_categorical_adjustment_multiarm_matches_modal_reference_dummies(method):
    """Each comparison's models fit on their own rows, so an arm where
    ``west`` outnumbers ``east`` takes ``west`` as its reference: the same
    column space as the oracle's dummies under a different parametrization,
    which the default learners' ridge penalty (1e-6 on standardized
    coefficients) separates by that order of magnitude, never more."""
    units = categorical_units(900, seed=5, arms=3)
    oracle = _rows(dummy_table(units), _DUMMIES, method, overlap="trim")
    actual = _rows(raw_table(units), _RAW, method, overlap="trim")
    assert [row["group_id"] for row in actual] == ["T1", "T2"]
    assert_rows_match(oracle, actual, rel=1e-6)


@pytest.mark.parametrize("method", _METHODS)
def test_categorical_adjustment_is_row_order_and_label_invariant(method):
    units = categorical_units(600, seed=3)
    baseline = _rows(raw_table(units), _RAW, method, overlap="trim")
    order = np.random.default_rng(1).permutation(len(units["user_id"]))
    shuffled = {name: values[order] for name, values in units.items()}
    assert_rows_match(baseline, _rows(raw_table(shuffled), _RAW, method, overlap="trim"), rel=1e-6)
    # Labels are names, not values: renaming every level (and with it the
    # lexical order of the labels) leaves every number where it was.
    relabelled = dict(units)
    relabelled["region"] = np.array(
        [{"east": "Zone E", "west": "Zone W", "north": "Alpha"}[r] for r in units["region"]]
    )
    assert_rows_match(baseline, _rows(raw_table(relabelled), _RAW, method, overlap="trim"))


@pytest.mark.parametrize("method", _METHODS)
def test_single_level_categorical_is_accepted_like_a_constant_column(method):
    units = categorical_units(400, seed=7)
    units["region"] = np.array(["east"] * len(units["region"]))
    without = _rows(raw_table(units), ("spend",), method, overlap="trim")
    with_constant = _rows(raw_table(units), _RAW, method, overlap="trim")
    assert_rows_match(without, with_constant)


@pytest.mark.parametrize("method", _METHODS)
def test_categorical_null_refuses_under_the_default_missing_policy(method):
    table = with_nulls(raw_table(categorical_units(400, seed=7)), "region", [3, 17, 40])
    with pytest.raises(IdentificationError) as exc_info:
        estimate_ate(_cat_source(table), _cat_design(_RAW), methods=[Method(name=method)])
    assert exc_info.value.code == "adjust.identification.missing_covariates"
    assert exc_info.value.context["missing_covariates"] == (("region", 3),)


@pytest.mark.parametrize("method", _METHODS)
def test_categorical_null_complete_case_matches_dropping_the_rows(method):
    units = categorical_units(600, seed=3)
    nulled = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    table = with_nulls(raw_table(units), "region", nulled)
    kept = {name: values[10:] for name, values in units.items()}
    oracle = _rows(dummy_table(kept), _DUMMIES, method, overlap="trim")
    actual = _rows(table, _RAW, method, missing="complete-case", overlap="trim")
    assert actual[0]["population"].startswith("complete cases (590 of 600 units)")
    assert_rows_match(oracle, actual, skip=("population",))


@pytest.mark.parametrize("method", _METHODS)
def test_categorical_null_impute_indicator_matches_modal_imputed_dummies_with_indicator(method):
    units = categorical_units(600, seed=3)
    nulled = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    table = with_nulls(raw_table(units), "region", nulled)
    # The oracle: a null level imputed with the modal level (all dummies 0)
    # plus an explicit indicator column entering the adjustment set.
    imputed = dict(units)
    imputed["region"] = units["region"].copy()
    imputed["region"][nulled] = "east"
    oracle_table = dummy_table(imputed).append_column(
        "region__missing", pa.array([1.0 if i in nulled else 0.0 for i in range(600)])
    )
    oracle = _rows(oracle_table, (*_DUMMIES, "region__missing"), method, overlap="trim")
    actual = _rows(table, _RAW, method, missing="impute-indicator", overlap="trim")
    assert_rows_match(oracle, actual, skip=("note",))


def test_categorical_null_pattern_matches_dummies_with_nan():
    units = categorical_units(900, seed=3)
    nulled = list(range(40))
    table = with_nulls(raw_table(units), "region", nulled)
    oracle_table = dummy_table(units)
    for column in DUMMY_COLUMNS:
        oracle_table = with_nulls(oracle_table, column, nulled)
    oracle = _rows(oracle_table, _DUMMIES, "iptw", missing="pattern", overlap="trim")
    actual = _rows(table, _RAW, "iptw", missing="pattern", overlap="trim")
    assert "2 patterns" in actual[0]["note"]
    assert_rows_match(oracle, actual)


class _MeanImputingLogistic:
    """A NaN-native stand-in: column-mean imputes NaN, then fits the default
    logistic model, so the fitted propensity genuinely reads every column."""

    def __init__(self):
        self._inner = LogisticPropensity()
        self._means = None

    def _filled(self, X):
        assert self._means is not None
        X = np.asarray(X, dtype=float)
        filled = X.copy()
        rows, cols = np.nonzero(np.isnan(filled))
        filled[rows, cols] = self._means[cols]
        return filled

    def fit(self, X, d):
        X = np.asarray(X, dtype=float)
        self._means = np.nan_to_num(np.nanmean(X, axis=0))
        self._inner.fit(self._filled(X), d)

    def predict(self, X):
        return self._inner.predict(self._filled(X))


def test_categorical_null_allow_matches_dummies_with_nan():
    units = categorical_units(600, seed=3)
    nulled = list(range(30))
    table = with_nulls(raw_table(units), "region", nulled)
    oracle_table = dummy_table(units)
    for column in DUMMY_COLUMNS:
        oracle_table = with_nulls(oracle_table, column, nulled)

    def run(source_table, covariates):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = estimate_ate(
                _cat_source(source_table),
                _cat_design(covariates, missing="allow", overlap="trim"),
                methods=[Method(name="iptw", propensity_learner=_MeanImputingLogistic)],
            ).results
        return [row.model_dump() for row in results]

    oracle = run(oracle_table, _DUMMIES)
    actual = run(table, _RAW)
    assert_rows_match(oracle, actual, skip=("note",))


def test_categorical_balance_diagnostics_name_levels_with_the_dummy_smds():
    """Every non-reference level is one SMD row named ``region=<level>``,
    carrying the value the dummy column would have: the balance gate sees
    a categorical exactly as it sees hand-built dummies."""
    units = categorical_units(300, seed=9)
    table = raw_table(units)

    def stub():
        return StubLearner(np.full(300, 0.5))

    def exceeded(source_table, covariates):
        with pytest.raises(IdentificationError) as exc_info:
            estimate_ate(
                _cat_source(source_table),
                _cat_design(covariates, max_smd=0.01),
                methods=[Method(name="iptw", propensity_learner=stub)],
            )
        assert exc_info.value.code == "adjust.identification.balance_gate_exceeded"
        entries = exc_info.value.context["exceeded"]
        assert isinstance(entries, tuple)
        return dict(cast("tuple[tuple[str, float], ...]", entries))

    oracle = exceeded(dummy_table(units), _DUMMIES)
    actual = exceeded(table, _RAW)
    assert set(actual) == {"spend", "region=west", "region=north"}
    assert actual["region=west"] == pytest.approx(oracle["region_west"], rel=1e-12)
    assert actual["region=north"] == pytest.approx(oracle["region_north"], rel=1e-12)
    assert actual["spend"] == pytest.approx(oracle["spend"], rel=1e-12)


def test_categorical_adjustment_hands_custom_learners_the_encoded_design():
    """A caller's own learner receives the float design a dummy encoding
    would have given it -- numeric columns first, then one indicator per
    non-reference level -- with NaN nowhere and no code column."""
    units = categorical_units(300, seed=9)
    seen: list[np.ndarray] = []

    class Recording:
        def fit(self, X, d):
            seen.append(np.asarray(X, dtype=float).copy())
            self._p = float(np.mean(d))

        def predict(self, X):
            seen.append(np.asarray(X, dtype=float).copy())
            return np.full(np.asarray(X).shape[0], self._p)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimate_ate(
            _cat_source(raw_table(units)),
            _cat_design(_RAW),
            methods=[Method(name="iptw", propensity_learner=Recording)],
        )
    train, test = seen
    assert train.shape == (300, 3) and test.shape == (300, 3)
    np.testing.assert_array_equal(train, test)
    np.testing.assert_array_equal(train[:, 0], units["spend"])
    np.testing.assert_array_equal(train[:, 1], (units["region"] == "west").astype(float))
    np.testing.assert_array_equal(train[:, 2], (units["region"] == "north").astype(float))


def test_categorical_adjustment_refuses_a_dtype_it_cannot_adjust_on():
    import datetime as dt

    units = categorical_units(60, seed=2)
    table = raw_table(units).append_column(
        "joined", pa.array([dt.date(2024, 1, 1 + i % 28) for i in range(60)])
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_ate(_cat_source(table), _cat_design(("spend", "joined")))
    assert exc_info.value.code == "estimation.adjust_common.covariate_dtype"
    assert exc_info.value.context["covariate"] == "joined"


def _unseen_level_advisories(caught) -> list[IncrementWarning]:
    return [
        w.message
        for w in caught
        if getattr(w.message, "code", None) == "estimation.adjust_common.unseen_level_advisory"
    ]


def test_iptw_discloses_a_level_only_another_treatment_carries():
    """Each comparison's propensity fits on its own treatment and the
    control, then predicts every cohort row: a level held by one ``T2``
    unit alone is absent from the ``T1`` fit's training rows, which scores
    that row as its reference level. The advisory names the one row and
    that one fit -- the ``T2`` fit trained on it and adds nothing -- and
    every contrast's note carries the disclosure. The binary cohort, whose
    single in-sample fit sees every row, discloses nothing."""

    def recorded(units):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = estimate_ate(
                _cat_source(raw_table(units)),
                _cat_design(_RAW, overlap="trim"),
                methods=[Method(name="iptw")],
            ).results
        return results, _unseen_level_advisories(caught)

    units = categorical_units(900, seed=5, arms=3)
    units["region"] = units["region"].astype(object)
    holder = int(np.flatnonzero(units["variant"] == "T2")[0])
    units["region"][holder] = "island"
    results, (advisory,) = recorded(units)
    assert advisory.context["unseen"] == (("region", "island", 1, 1),)
    assert advisory.context["n"] == 900
    assert advisory.context["treatment_groups"] == ("T1", "T2")
    assert [row.group_id for row in results] == ["T1", "T2"]

    binary = categorical_units(300, seed=21)
    binary["region"] = binary["region"].astype(object)
    binary["region"][5] = "island"
    _, advisories = recorded(binary)
    assert advisories == []


class _NanIndicatorLogistic:
    """A NaN-native stand-in whose fit genuinely depends on the NaN pattern:
    zero-fills NaN and appends one is-missing indicator per column that
    carried NaN in training, then fits the default logistic model."""

    def __init__(self):
        self._inner = LogisticPropensity()
        self._nan_cols = None

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.nan_to_num(X), np.isnan(X[:, self._nan_cols]).astype(float)])

    def fit(self, X, d):
        X = np.asarray(X, dtype=float)
        self._nan_cols = np.flatnonzero(np.isnan(X).any(axis=0))
        self._inner.fit(self._design(X), d)

    def predict(self, X):
        return self._inner.predict(self._design(X))


def test_allow_single_level_categorical_hands_its_null_rows_to_the_learner_as_nan():
    """Under missing='allow' a categorical whose observed rows hold one
    level has no non-reference indicator, yet its null rows must still
    reach the NaN-native learner: the fitted design carries one all-zero
    column that is NaN exactly on those rows, in the fit, in the prediction
    and in the capability probe alike. The estimate then matches the oracle
    a user builds by carrying the null level in one numeric column that is
    constant where observed -- a learner that reads the NaN pattern sees
    the same design either way."""
    units = categorical_units(600, seed=3)
    units["region"] = np.array(["east"] * 600)
    nulled = list(range(30))
    table = with_nulls(raw_table(units), "region", nulled)
    seen: list[np.ndarray] = []

    class Recording(_NanIndicatorLogistic):
        def fit(self, X, d):
            seen.append(np.asarray(X, dtype=float).copy())
            super().fit(X, d)

        def predict(self, X):
            seen.append(np.asarray(X, dtype=float).copy())
            return super().predict(X)

    def run(source_table, covariates, learner):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = estimate_ate(
                _cat_source(source_table),
                _cat_design(covariates, missing="allow", overlap="trim"),
                methods=[Method(name="iptw", propensity_learner=learner)],
            ).results
        return [row.model_dump() for row in results]

    actual = run(table, _RAW, Recording)
    train, test = (X for X in seen if X.shape[0] == 600)
    assert train.shape == (600, 2)
    np.testing.assert_array_equal(train, test)
    np.testing.assert_array_equal(train[:, 0], units["spend"])
    null = np.zeros(600, dtype=bool)
    null[nulled] = True
    np.testing.assert_array_equal(np.isnan(train[:, 1]), null)
    np.testing.assert_array_equal(train[~null, 1], 0.0)
    probes = [X for X in seen if X.shape[0] != 600]
    assert probes and all(X.shape[1] == 2 for X in probes)
    assert np.isnan(probes[0][:, 1]).any() and not np.isnan(probes[0][:, 0]).any()

    carrier = pa.array([None if i in nulled else 0.0 for i in range(600)], type=pa.float64())
    oracle_units = {name: values for name, values in units.items() if name != "region"}
    oracle_table = raw_table(oracle_units).append_column("region_null", carrier)
    oracle = run(oracle_table, ("spend", "region_null"), _NanIndicatorLogistic)
    assert_rows_match(oracle, actual, skip=("note",))
