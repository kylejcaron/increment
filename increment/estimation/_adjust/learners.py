"""Learner protocols and default nuisance models for observational adjustment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from increment.errors import InvalidRequestError, raiser, refusals

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.adjust_learners.logistic_propensity.logisticpropensity_failed_converge": "LogisticPropensity failed to converge to a finite solution: {res}",
        "estimation.adjust_learners.logistic_propensity.logisticpropensity_predict_called": "LogisticPropensity.predict called before fit",
        "estimation.adjust_learners.ridge_outcome.ridgeoutcome_predict_called": "RidgeOutcome.predict called before fit",
    },
)
_raise = raiser(_REFUSALS)


class Learner(Protocol):
    """A propensity/outcome model: `fit` on covariates + label, then `predict`.

    Stateless protocol (not ABC): any object satisfying the interface
    works, including a user's own sklearn/LightGBM wrapper; no
    inheritance required.
    """

    def fit(self, X: np.ndarray, d: np.ndarray) -> None:
        """Fit the model on covariate matrix `X` and binary label `d`."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities in `[0, 1]` for each row of `X`.

        The closed interval, not open: a learner's sigmoid can saturate to
        exactly 0.0/1.0 for extreme inputs. Callers relying on `1/e` or
        `1/(1-e)` staying finite must gate on this via an overlap check
        (e.g. `IdentificationGate.min_propensity`) rather than assume the
        open interval.
        """
        ...


# Concrete Learner: scipy-only ridge-regularized logistic regression


class LogisticPropensity:
    """Ridge-regularized logistic regression fit via `scipy.optimize.minimize`.

    Implements `Learner`. Columns of `X` are standardized internally before
    fitting (a zero-variance column gets standard deviation 1.0 instead of
    dividing by zero); `predict` re-applies the stored standardization.
    The intercept is not penalized. `l2` is the ridge penalty weight on the
    standardized-scale coefficients (excluding the intercept).
    """

    def __init__(self, l2: float = 1e-6) -> None:
        self._l2 = l2
        self._mean: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._coef: np.ndarray | None = None

    def fit(self, X: np.ndarray, d: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        d = np.asarray(d, dtype=float)

        mean = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        Xs = (X - mean) / sd

        n, k = Xs.shape
        design = np.column_stack([np.ones(n), Xs])
        l2 = self._l2

        def nll_and_grad(b: np.ndarray) -> tuple[float, np.ndarray]:
            eta = design @ b
            nll = np.logaddexp(0.0, eta).sum() - d @ eta + l2 * (b[1:] @ b[1:])
            grad = design.T @ (expit(eta) - d)
            grad[1:] += 2.0 * l2 * b[1:]
            return nll, grad

        res = minimize(
            nll_and_grad,
            x0=np.zeros(k + 1),
            jac=True,
            method="L-BFGS-B",
        )
        # Gate on solution quality, not `res.success`: L-BFGS-B often reports
        # success=False on near-separable fits while `res.x` is still usable.
        if not np.isfinite(res.x).all():
            _raise(
                "estimation.adjust_learners.logistic_propensity.logisticpropensity_failed_converge",
                res=res.message,
            )

        self._mean = mean
        self._sd = sd
        self._coef = res.x

    def _fitted_design(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Intercept plus the stored standardization of `X`, and the coefficients."""
        if self._coef is None or self._mean is None or self._sd is None:
            _raise(
                "estimation.adjust_learners.logistic_propensity.logisticpropensity_predict_called"
            )
        X = np.asarray(X, dtype=float)
        Xs = (X - self._mean) / self._sd
        n = Xs.shape[0]
        return np.column_stack([np.ones(n), Xs]), self._coef

    def predict(self, X: np.ndarray) -> np.ndarray:
        design, coef = self._fitted_design(X)
        return expit(design @ coef)


def _coupled_hajek_mean_influences(
    models: Sequence[LogisticPropensity],
    designs: Sequence[np.ndarray],
    arm: np.ndarray,
    conditional: np.ndarray,
    marginal: np.ndarray,
    fixed: np.ndarray,
) -> np.ndarray:
    """Joint coupled-logistic/Hajek estimating-equation arm-mean influences.

    ``models[b - 1]`` is the fitted logit of arm ``b`` against control on
    their rows only and ``designs[b - 1]`` the covariate matrix it was
    fitted on, evaluated on every cohort row (one matrix per model, since a
    model's level encoding is fitted on its own rows); ``conditional[:, b -
    1]`` is its prediction q_b on every cohort row, ``marginal`` the coupled
    arm propensities (control column 0) and ``fixed`` each arm's
    fixed-propensity Hajek influence (N w_t/W_t)(Y - mu_t), one column per
    arm.

    With S_b the pair rows, D_b = 1{A = b} and Z_b the fitted standardized
    design evaluated on every row, model b's score s_b = S_b Z_b (D_b - q_b)
    is centered over all N rows and H_b = mean[S_b q_b (1 - q_b) Z_b Z_b'] plus
    its penalty Hessian over N. Coupling gives d log p_t / d beta_b =
    (1{t = b} - p_b) Z_b, so arm t's influence gains
    -mean[fixed_t (1{t = b} - p_b) Z_b]' H_b^+ s_b for every model b. The
    models are fitted separately (block-diagonal Jacobian) but share control
    rows; keeping every row's complete corrected influence retains that
    cross-model covariance. With one treatment this is the binary joint
    logistic/Hajek correction. It is an asymptotic sandwich conditional on
    each design transform; a fixed ridge penalty vanishes relative to sample
    information.
    """
    n = fixed.shape[0]
    binary = len(models) == 1
    arms = np.arange(fixed.shape[1])
    corrected = fixed.copy()
    for b, (model, X) in enumerate(zip(models, designs, strict=True), start=1):
        design, _ = model._fitted_design(X)
        q = conditional[:, b - 1]
        treated = (arm == b).astype(float)
        curvature = q * (1.0 - q)
        residual = treated - q
        if not binary:
            in_pair = ((arm == 0) | (arm == b)).astype(float)
            curvature = in_pair * curvature
            residual = in_pair * residual
        penalty = 2.0 * model._l2 * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        hessian = (design.T @ (curvature[:, None] * design) + penalty) / n
        score = design * residual[:, None]
        score -= score.mean(axis=0)
        lever = (arms == b)[None, :] - marginal[:, b][:, None]
        derivative = -design.T @ (fixed * lever) / n
        # A redundant constant covariate does not change the fitted score space.
        inverse = np.linalg.pinv(hessian, hermitian=True)
        corrected += score @ (inverse @ derivative)
    return corrected


class RidgeOutcome:
    """Closed-form ridge linear regression: the outcome-model half of DML.

    Implements `Learner` structurally; `predict` returns conditional means
    (unbounded), not probabilities. Columns of `X` are standardized
    internally before fitting; the intercept is not penalized. Fit is
    `np.linalg.solve` on the ridge normal equations, no iterative
    optimizer or convergence failure mode. `l2` is the ridge penalty on
    standardized-scale coefficients (excluding the intercept).
    """

    def __init__(self, l2: float = 1e-6) -> None:
        self._l2 = l2
        self._mean: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._coef: np.ndarray | None = None

    def fit(self, X: np.ndarray, d: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(d, dtype=float)
        mean = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        Xs = (X - mean) / sd
        n, k = Xs.shape
        design = np.column_stack([np.ones(n), Xs])
        penalty = 2.0 * self._l2 * np.eye(k + 1)
        penalty[0, 0] = 0.0  # unpenalized intercept
        self._coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        self._mean = mean
        self._sd = sd

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coef is None or self._mean is None or self._sd is None:
            _raise("estimation.adjust_learners.ridge_outcome.ridgeoutcome_predict_called")
        X = np.asarray(X, dtype=float)
        Xs = (X - self._mean) / self._sd
        design = np.column_stack([np.ones(Xs.shape[0]), Xs])
        return design @ self._coef
