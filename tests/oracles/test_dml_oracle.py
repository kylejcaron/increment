"""Cross-checks increment's iid DML/AIPW/IPTW point and SE against a
frozen doubleml PLR (DML)/doubleml IRM (AIPW)/statsmodels WLS-HC0 (IPTW)
oracle fed the identical out-of-fold nuisance predictions (see
tests/oracles/generate/gen_dml_oracle.py). No doubleml/statsmodels/
scikit-learn at test time -- the frozen nuisances are looked up by
covariate row, not refit."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from increment.estimation.adjust import estimate_ate
from increment.estimation.engine import Method
from increment.frame import from_unit_summary
from increment.semantics.design import AdjustmentSet, Observational

_HERE = Path(__file__).parent
_FIXTURE = json.loads((_HERE / "fixtures" / "dml_aipw_iptw.json").read_text())
_DF = pd.read_csv(_HERE / "data" / "dml_synthetic.csv")
_COVARIATES = ("x1", "x2", "x3")
_TOL = _FIXTURE["tolerance"]


class _FrozenLearner:
    """Test-only Learner returning a precomputed out-of-fold prediction,
    looked up by covariate row rather than fit at test time."""

    def __init__(self, predictions: list[float]):
        rows = _DF[list(_COVARIATES)].to_numpy()
        self._lookup = {tuple(row): float(p) for row, p in zip(rows, predictions, strict=True)}

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.array([self._lookup[tuple(row)] for row in X])


def _alternating_outcome_factory(treat_predictions: list[float], control_predictions: list[float]):
    """AIPW's public `outcome_learner` argument is a single factory reused
    for every per-arm outcome model (see aipw_estimate in
    increment/estimation/_adjust/aipw.py). For one treatment,
    `_crossfit_nuisances` instantiates the treatment-arm model before the
    control-arm model in every fold -- so a factory that alternates between
    a treat-arm and control-arm `_FrozenLearner` on each call routes m1_hat
    to the treatment arm and m0_hat to the control arm on every fold."""
    learners = itertools.cycle(
        [_FrozenLearner(treat_predictions), _FrozenLearner(control_predictions)]
    )
    return lambda: next(learners)


def _src():
    tbl = _DF.rename(columns={"variant": "group"})
    return from_unit_summary(
        tbl,
        unit="unit_id",
        group="group",
        control="C",
        metrics={"y": "mean"},
    )


def _metric(src):
    return next(m for m in src.context.metrics if m.name == "y")


def _design():
    return Observational(control_group="C", adjustment=AdjustmentSet(covariates=_COVARIATES))


def test_dml_matches_doubleml_external_predictions():
    src = _src()
    (est,) = estimate_ate(
        src,
        _design(),
        methods=[
            Method(
                name="dml",
                propensity_learner=lambda: _FrozenLearner(_FIXTURE["nuisances"]["e_hat"]),
                outcome_learner=lambda: _FrozenLearner(_FIXTURE["nuisances"]["m_hat_pooled"]),
                folds=_FIXTURE["folds"],
            )
        ],
        metrics=[_metric(src)],
        value_scale={"y": "absolute"},
    ).results
    ref = _FIXTURE["dml"]
    assert est.lift is not None
    assert est.lift.value == pytest.approx(ref["theta"], rel=_TOL["dml_theta_rel"], abs=0.0)
    # log_se carries the raw pre-prior SE for whichever value_scale was
    # requested (see Estimate.log_mean/log_se docstring), not a log-scale-only field.
    assert est.lift.log_se == pytest.approx(ref["se"], rel=_TOL["dml_se_rel"], abs=0.0)


def test_aipw_matches_doubleml_irm_ate_score():
    src = _src()
    (est,) = estimate_ate(
        src,
        _design(),
        methods=[
            Method(
                name="aipw",
                propensity_learner=lambda: _FrozenLearner(_FIXTURE["nuisances"]["e_hat"]),
                outcome_learner=_alternating_outcome_factory(
                    _FIXTURE["nuisances"]["m1_hat"], _FIXTURE["nuisances"]["m0_hat"]
                ),
                folds=_FIXTURE["folds"],
            )
        ],
        metrics=[_metric(src)],
        value_scale={"y": "absolute"},
    ).results
    ref = _FIXTURE["aipw"]
    assert est.lift is not None
    assert est.lift.value == pytest.approx(ref["theta"], rel=_TOL["aipw_theta_rel"], abs=0.0)
    assert est.lift.log_se == pytest.approx(ref["se"], rel=_TOL["aipw_se_rel"], abs=0.0)


def test_iptw_matches_statsmodels_wls_hajek():
    src = _src()
    (est,) = estimate_ate(
        src,
        _design(),
        methods=[
            Method(
                name="iptw",
                propensity_learner=lambda: _FrozenLearner(_FIXTURE["nuisances"]["e_hat"]),
            )
        ],
        metrics=[_metric(src)],
        value_scale={"y": "absolute"},
    ).results
    ref = _FIXTURE["iptw"]
    assert est.lift is not None
    assert est.lift.value == pytest.approx(ref["theta"], rel=_TOL["iptw_theta_rel"], abs=0.0)
    assert est.lift.log_se == pytest.approx(ref["se"], rel=_TOL["iptw_se_rel"], abs=0.0)
