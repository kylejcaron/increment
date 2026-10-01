"""Step 2: Failing Learner protocol + LogisticPropensity + ADJUSTMENTS tests."""

from typing import cast

import numpy as np
import pytest
from scipy.special import expit

from increment.errors import CodedError, InvalidRequestError, UnsupportedRequestError
from increment.estimation._adjust.learners import LogisticPropensity
from increment.estimation.adjust import ADJUSTMENTS


def test_logistic_propensity_recovers_signal():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(4000, 2))
    p = expit(0.4 + 1.5 * X[:, 0] - 1.0 * X[:, 1])
    d = rng.binomial(1, p).astype(float)
    lr = LogisticPropensity()
    lr.fit(X, d)
    e = lr.predict(X)
    assert e.shape == (4000,)
    assert np.all((e > 0) & (e < 1))
    assert np.corrcoef(e, p)[0, 1] > 0.97  # fitted probabilities track the truth


def test_logistic_propensity_constant_column_does_not_crash():
    X = np.column_stack([np.ones(200), np.linspace(-1, 1, 200)])
    d = (np.linspace(-1, 1, 200) > 0).astype(float)
    lr = LogisticPropensity()
    lr.fit(X, d)
    assert np.all(np.isfinite(lr.predict(X)))


def test_registry_unknown_adjustment_names_available_keys():
    with pytest.raises(UnsupportedRequestError) as exc_info:
        ADJUSTMENTS.get("definitely-not-registered")
    assert exc_info.value.code == "estimation.variance.registry.no_registered_available"
    assert exc_info.value.context["key"] == "definitely-not-registered"
    assert {"iptw", "aipw", "dml"} <= set(cast("list[str]", exc_info.value.context["available"]))


def test_logistic_propensity_predict_before_fit_raises():
    lr = LogisticPropensity()
    with pytest.raises(InvalidRequestError) as exc_info:
        lr.predict(np.zeros((3, 2)))
    assert (
        exc_info.value.code
        == "estimation.adjust_learners.logistic_propensity.logisticpropensity_predict_called"
    )


def test_logistic_propensity_gates_on_finite_solution_not_success_flag(monkeypatch):
    """L-BFGS-B can report success=False on near-separable logistic problems
    while `res.x` is still finite and usable. fit() must gate on solution
    finiteness, not the unreliable `success` flag, so the optimizer result
    is faked directly here to exercise that gating logic."""
    import increment.estimation._adjust.learners as learners_module

    class _FakeResult:
        success = False
        status = 2
        message = "ABNORMAL_TERMINATION_IN_LNSRCH"
        x = np.array([0.1, 0.5])  # finite, usable

    monkeypatch.setattr(learners_module, "minimize", lambda *a, **k: _FakeResult())

    lr = LogisticPropensity()
    lr.fit(np.zeros((10, 1)), np.array([1.0, 0.0] * 5))  # must not raise
    assert np.array_equal(lr.predict(np.zeros((3, 1))).shape, (3,))


def test_logistic_propensity_raises_on_nonfinite_solution(monkeypatch):
    """The flip side of the gate above: a genuinely non-finite solution
    must still raise, regardless of what `success` claims."""
    import increment.estimation._adjust.learners as learners_module

    class _FakeResult:
        success = True  # even a "successful" optimizer run with NaN coefficients
        status = 0
        message = "CONVERGENCE"
        x = np.array([np.nan, 0.5])

    monkeypatch.setattr(learners_module, "minimize", lambda *a, **k: _FakeResult())

    lr = LogisticPropensity()
    with pytest.raises(InvalidRequestError) as exc_info:
        lr.fit(np.zeros((10, 1)), np.array([1.0, 0.0] * 5))
    assert (
        exc_info.value.code
        == "estimation.adjust_learners.logistic_propensity.logisticpropensity_failed_converge"
    )


class _CountingLearner:
    """A Learner whose per-fit state would leak if the instance were shared."""

    instances: list["_CountingLearner"] = []

    def __init__(self) -> None:
        _CountingLearner.instances.append(self)
        self.fits = 0

    def fit(self, X, d) -> None:
        self.fits += 1

    def predict(self, X):
        import numpy as np

        return np.full(X.shape[0], 0.5)


def test_iptw_builds_a_fresh_propensity_learner_for_every_contrast():
    """Method.propensity_learner is a factory for all three adjustments;
    DML/AIPW honour that per fold per role, IPTW must too."""
    import warnings

    import numpy as np
    import pyarrow as pa

    from increment.estimation.adjust import estimate_ate
    from increment.estimation.engine import Method
    from increment.frame import from_unit_summary
    from increment.semantics.design import AdjustmentSet, IdentificationGate, Observational

    _CountingLearner.instances.clear()
    n, rng = 400, np.random.default_rng(5)
    x = rng.normal(size=n)
    arm = np.where(rng.random(n) < 0.34, "C", np.where(rng.random(n) < 0.5, "T", "T2"))
    src = from_unit_summary(
        pa.table(
            {
                "user_id": [f"u{i}" for i in range(n)],
                "variant": arm,
                "revenue": 1.0 + x + rng.normal(size=n),
                "other": 2.0 + x + rng.normal(size=n),
                "x": x,
            }
        ),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean", "other": "mean"},
    )
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x",)),
        gate=IdentificationGate(max_smd=5.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = estimate_ate(
            src, design, methods=[Method(name="iptw", propensity_learner=_CountingLearner)]
        )

    assert len(out.results) == 4  # 2 metrics x 2 treatment arms
    assert len(_CountingLearner.instances) == 4
    assert [inst.fits for inst in _CountingLearner.instances] == [1, 1, 1, 1]


class ScalarLearner:
    """A malformed learner: predict returns a scalar, broadcasting against
    every row instead of raising."""

    def fit(self, X, d):
        return self

    def predict(self, X):
        return 0.5


def _confounded_source(n=400, seed=3):
    import pyarrow as pa
    from scipy.special import expit

    from increment.frame import from_unit_summary

    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = rng.binomial(1, expit(1.5 * x))
    y = 3.0 * x + 2.0 * d + 0.1 * rng.normal(size=n)
    return from_unit_summary(
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


@pytest.mark.parametrize("method_name", ["iptw", "aipw", "dml"])
def test_scalar_prediction_refuses_with_prediction_shape_code(method_name):
    from increment.estimation.adjust import estimate_ate
    from increment.estimation.engine import Method
    from increment.semantics.design import AdjustmentSet, Observational

    src = _confounded_source()
    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
    if method_name == "iptw":
        method = Method(name=method_name, propensity_learner=ScalarLearner)
    else:
        method = Method(
            name=method_name, propensity_learner=ScalarLearner, outcome_learner=ScalarLearner
        )
    with pytest.raises(CodedError) as exc_info:
        estimate_ate(src, design, methods=[method])
    assert exc_info.value.code == "estimation.adjust.learner.prediction_shape"


def _pattern_confounded_source(n=400, seed=11):
    import pyarrow as pa

    from increment.frame import from_unit_summary

    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    miss = rng.random(n) < 0.3
    d = rng.binomial(1, expit(1.2 * x + 0.8 * z))
    y = 2.0 * z + 3.0 * x + 2.0 * d + 0.1 * rng.normal(size=n)
    return from_unit_summary(
        pa.table(
            {
                "user_id": [f"u{i}" for i in range(n)],
                "variant": np.where(d == 1, "T", "C"),
                "revenue": y,
                "x": np.where(miss, np.nan, x),
                "z": z,
            }
        ),
        unit="user_id",
        group="variant",
        control="C",
        metrics={"revenue": "mean"},
    )


def test_pattern_mode_scalar_prediction_refuses_with_prediction_shape_code():
    from increment.estimation.adjust import estimate_ate
    from increment.estimation.engine import Method
    from increment.semantics.design import AdjustmentSet, Observational

    src = _pattern_confounded_source()
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet(covariates=("x", "z"), missing="pattern"),
    )
    with pytest.raises(CodedError) as exc_info:
        estimate_ate(src, design, methods=[Method(name="iptw", propensity_learner=ScalarLearner)])
    assert exc_info.value.code == "estimation.adjust.learner.prediction_shape"
