"""Generate the 500-row synthetic dataset and the doubleml/statsmodels
oracle values for increment's iid DML, AIPW, and IPTW adjustment
estimators. Run once; commits tests/oracles/data/dml_synthetic.csv and
tests/oracles/fixtures/dml_aipw_iptw.json.

Out-of-fold nuisance predictions are computed here (not re-derived at
test time) using increment's own deterministic hash-based fold
assignment (_fold_ids), so increment's estimators and the oracle tools
below all consume literally the same numbers -- the comparison isolates
each implementation's aggregation formula, not learner or fold-split
variance.

DML's oracle is doubleml's DoubleMLPLR (a genuinely independent
partially-linear-model implementation). AIPW's oracle is doubleml's
DoubleMLIRM ATE score (score="ATE", normalize_ipw=True) fed the same
frozen (e_hat, m1_hat, m0_hat) via external_predictions -- NOT a hand-
rolled restatement of increment's own augmentation formula, which would
let a shared bug in both sides go undetected. IPTW's oracle is
statsmodels' WLS (Hajek IPW) with an HC0 sandwich SE, a materially
different codebase that happens to reduce algebraically to the exact
same saturated-regression variance formula as increment's Hajek
ratio-estimator delta-method SE for this single-binary-regressor case
(measured to agree at ~1e-16 relative -- see tolerance.note).

Package versions this was verified against (recorded automatically into
the fixture below).

Usage (from the repository root, with doubleml/statsmodels/scikit-learn
installed -- these are oracle-generation-only tools, not project
dependencies):
    uv run --with doubleml --with statsmodels --with scikit-learn \\
        python tests/oracles/generate/gen_dml_oracle.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import sklearn  # ty: ignore[unresolved-import]  - oracle-generation-only, not a project dependency
import statsmodels  # ty: ignore[unresolved-import]  - oracle-generation-only, not a project dependency
import statsmodels.api as sm  # ty: ignore[unresolved-import]  - oracle-generation-only, not a project dependency
from sklearn.linear_model import (  # ty: ignore[unresolved-import]
    LinearRegression,
    LogisticRegression,
)

from increment.estimation._adjust.overlap import _fold_ids

DATA_OUT = "tests/oracles/data/dml_synthetic.csv"
FIXTURE_OUT = "tests/oracles/fixtures/dml_aipw_iptw.json"
FOLDS = 5
TRUE_ATE = 2.0
# Matches increment's own propensity clip: neither side's variance
# formula should be distorted by an oracle-only trimming policy.
PROPENSITY_CLIP = 1e-6

rng = np.random.default_rng(20260915)
n = 500
x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
propensity_true = 1.0 / (1.0 + np.exp(-(0.4 * x1 - 0.3 * x2 + 0.2 * x3)))
d = rng.binomial(1, propensity_true)
noise = rng.normal(scale=1.0, size=n)
y = 1.0 + 1.5 * x1 - 1.0 * x2 + 0.5 * x3 + TRUE_ATE * d + noise
unit_id = np.array([f"u{i:04d}" for i in range(n)])
variant = np.where(d == 1, "T", "C")

df = pd.DataFrame({"unit_id": unit_id, "variant": variant, "y": y, "x1": x1, "x2": x2, "x3": x3})
df.to_csv(DATA_OUT, index=False)

x_all = np.column_stack([x1, x2, x3])
folds = _fold_ids(unit_id.astype(str), variant.astype(str), FOLDS)


def _oof_propensity(x: np.ndarray, d_: np.ndarray, folds_: np.ndarray) -> np.ndarray:
    e_hat = np.empty(len(d_))
    for k in range(FOLDS):
        train = folds_ != k
        model = LogisticRegression(penalty=None, max_iter=2000).fit(x[train], d_[train])
        e_hat[~train] = model.predict_proba(x[~train])[:, 1]
    return np.clip(e_hat, PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)


def _oof_outcome(x: np.ndarray, y_: np.ndarray, folds_: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions for y_[mask], fitting only on mask & train."""
    m_hat = np.full(len(y_), np.nan)
    for k in range(FOLDS):
        train = (folds_ != k) & mask
        val = (folds_ == k) & mask
        if not train.any() or not val.any():
            continue
        model = LinearRegression().fit(x[train], y_[train])
        m_hat[val] = model.predict(x[val])
    return m_hat


e_hat = _oof_propensity(x_all, d, folds)
# DML's pooled outcome model: trained on the WHOLE fold (both arms).
m_hat_pooled = _oof_outcome(x_all, y, folds, mask=np.ones(n, dtype=bool))
# AIPW's per-arm outcome models: trained on each arm's fold subset only.
m1_hat = _oof_outcome(x_all, y, folds, mask=(d == 1))
m0_hat = _oof_outcome(x_all, y, folds, mask=(d == 0))
# AIPW needs a prediction for EVERY unit from both arm models (the
# counterfactual mean), so refit outside the per-arm fold restriction
# for whichever arm a unit was NOT observed in, still out-of-fold.
for k in range(FOLDS):
    train1 = (folds != k) & (d == 1)
    val_other = (folds == k) & (d == 0)
    model1 = LinearRegression().fit(x_all[train1], y[train1])
    m1_hat[val_other] = model1.predict(x_all[val_other])
    train0 = (folds != k) & (d == 0)
    val_other1 = (folds == k) & (d == 1)
    model0 = LinearRegression().fit(x_all[train0], y[train0])
    m0_hat[val_other1] = model0.predict(x_all[val_other1])

# --- doubleml PLR oracle for DML, fed the identical (e_hat, m_hat_pooled) ---
import doubleml as dml  # noqa: E402  # ty: ignore[unresolved-import]  - oracle-generation-only dependency
from doubleml.utils.propensity_score_processing import (  # noqa: E402  # ty: ignore[unresolved-import]
    PSProcessorConfig,
)

dml_data = dml.DoubleMLData(df.assign(d=d), y_col="y", d_cols="d", x_cols=["x1", "x2", "x3"])
plr = dml.DoubleMLPLR(
    dml_data, LinearRegression(), LogisticRegression(penalty=None, max_iter=2000), n_folds=FOLDS
)
plr.fit(
    external_predictions={"d": {"ml_l": m_hat_pooled.reshape(-1, 1), "ml_m": e_hat.reshape(-1, 1)}}
)
dml_theta, dml_se = float(plr.coef[0]), float(plr.se[0])

# --- doubleml IRM (ATE score) oracle for AIPW on the same (e_hat, m1_hat,
# m0_hat): an independent AIPW implementation. normalize_ipw=True matches
# increment's Hajek weighting; clipping_threshold=PROPENSITY_CLIP stops
# doubleml re-clipping the already-clipped e_hat to a different band.
irm = dml.DoubleMLIRM(
    dml_data,
    ml_g=LinearRegression(),
    ml_m=LogisticRegression(penalty=None, max_iter=2000),
    n_folds=FOLDS,
    score="ATE",
    normalize_ipw=True,
    ps_processor_config=PSProcessorConfig(clipping_threshold=PROPENSITY_CLIP),
)
irm.fit(
    external_predictions={
        "d": {
            "ml_g0": m0_hat.reshape(-1, 1),
            "ml_g1": m1_hat.reshape(-1, 1),
            "ml_m": e_hat.reshape(-1, 1),
        }
    }
)
aipw_theta, aipw_se = float(irm.coef[0]), float(irm.se[0])

# --- statsmodels WLS (Hajek IPW) oracle for IPTW, fed the identical e_hat ---
iptw_weights = np.where(d == 1, 1.0 / e_hat, 1.0 / (1.0 - e_hat))
design = sm.add_constant(d.astype(float))
wls_fit = sm.WLS(y, design, weights=iptw_weights).fit(cov_type="HC0")
iptw_theta, iptw_se = float(wls_fit.params[1]), float(wls_fit.bse[1])

out = {
    "generator": "tests/oracles/generate/gen_dml_oracle.py",
    "package_versions": {
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "doubleml": dml.__version__,
    },
    "n": n,
    "folds": FOLDS,
    "true_ate": TRUE_ATE,
    "nuisances": {
        "e_hat": e_hat.tolist(),
        "m_hat_pooled": m_hat_pooled.tolist(),
        "m1_hat": m1_hat.tolist(),
        "m0_hat": m0_hat.tolist(),
        "x1": x1.tolist(),
        "x2": x2.tolist(),
        "x3": x3.tolist(),
    },
    "dml": {"theta": dml_theta, "se": dml_se},
    "aipw": {"theta": aipw_theta, "se": aipw_se},
    "iptw": {"theta": iptw_theta, "se": iptw_se},
    "tolerance": {
        "dml_theta_rel": 1e-9,
        "dml_se_rel": 1e-9,
        "aipw_theta_rel": 1e-9,
        "aipw_se_rel": 1e-4,
        "iptw_theta_rel": 1e-9,
        "iptw_se_rel": 1e-9,
        "note": (
            "Measured after running this generator (not assumed). DML "
            "theta/se and IPTW theta/se agree with doubleml's PLR and "
            "statsmodels' WLS-HC0 respectively to ~1e-16 relative -- both "
            "are the same closed-form aggregation over the identical "
            "frozen nuisances/weights, just computed by an independent "
            "codebase, so 1e-9 is cross-platform (BLAS/summation-order) "
            "headroom over the observed floating-point-level agreement, "
            "not a real methodological tolerance. AIPW's oracle is "
            "doubleml's DoubleMLIRM (score='ATE', normalize_ipw=True), a "
            "genuinely independent implementation of the augmented-IPW "
            "ATE (not a restatement of increment's own formula): theta "
            "matches to ~1e-16 (same aggregation identity), but se has a "
            "small, real, measured ~7e-5 relative gap -- doubleml's "
            "normalize_ipw scales the propensity used inside the score "
            "but does not apply increment's additional ratio-estimator "
            "delta-method correction to the outcome-model baseline term, "
            "so the two SE formulas are asymptotically equivalent but not "
            "algebraically identical; 1e-4 keeps headroom over the "
            "measured gap while remaining far tighter than any real "
            "formula defect (order 0.01-1) would produce."
        ),
    },
}
with open(FIXTURE_OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {DATA_OUT} and {FIXTURE_OUT}")
