"""Issue #73: does a nonlinear R-learner earn its place in treatment targeting?

Run with ``python -m calibration.nonlinear_targeting``. This is a prespecified
evidence harness, not a production route: it compares the shipped interacted
regression (:func:`increment.estimation.cate.fit_cate`) with one candidate,
an R-learner over a boosted base regressor (:mod:`calibration._rlearner`), and
ends in an implement/defer/reject recommendation. It adds no public API.

Workflow gate. The study assumes a named deployment decision (treat a unit
when its predicted effect exceeds a per-unit ``cost``), strictly pre-treatment
covariates, predeclared audit subgroups and correction cells, and enough
independent units or clusters to leave an evaluation split untouched.

Identification. Assignment is randomized at the unit with a constant treated
share; dependence is confined to declared clusters or repeated unit ids (no
interference across clusters, no carryover between a unit's rows); the
deployment population is the experiment population. Under these assumptions
the held-out score ``psi = (y - m(x)) (d - p) / (p (1 - p))``, with ``m``
fitted only on the fit split, has conditional mean ``tau(x)`` for any ``m``.

Splits. Units (or whole clusters) go to fit / selection / evaluation via
:func:`increment.estimation.crossfit.outer_split`, and every split is checked
with :func:`~increment.estimation.crossfit.check_cluster_atomic`. Learners and
the evaluation control variate are fitted on the fit split; the R-learner round
count, the simple treat-all/treat-none choice and every group-bias correction
are chosen on the selection split; every reported number comes from the
untouched evaluation split.

Inference. Policy values, paired value differences and subgroup discrepancies
are means of per-row contributions with cluster-summed (or unit-summed)
sandwich variances. Subgroup discrepancies ``delta_g = mean_g(tau_hat - psi)``
(group bias, as in Persson et al. 2026, arXiv:2602.20383) are reported jointly:
overlapping groups share a covariance, simultaneous intervals use the max-|z|
critical value of that covariance, and a group without enough evidence in
either arm is an unavailable row with its reason, outside the joint family.
The two primary policy comparisons (candidate versus current, candidate versus
simple) are an intersection-union test, so neither needs a multiplicity
adjustment. The R-learner carries no interval of its own.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm

from calibration._rlearner import BoostedRegressor, RLearner
from increment.errors import CodedError
from increment.estimation.cate import Covariate, fit_cate
from increment.estimation.crossfit import check_cluster_atomic, outer_split

ALPHA = 0.05
# Monte Carlo slack on coverage-type checks; at 100 replicates the binomial
# SE of a 0.95 rate is about 0.022.
MC_TOLERANCE = 0.05
# A material gain closes at least this share of the gap between the current
# learner's true policy value and the oracle's.
MATERIAL_GAP_SHARE = 0.25
# Tolerated true-value loss against the current learner where it is correctly
# specified, as a share of the oracle value.
HARM_SHARE = 0.10
MIN_ARM_UNITS = 50
MIN_ARM_CLUSTERS = 20
REFERENCE_N = 100_000
SPLIT_SHARES = {"fit": 0.4, "select": 0.3, "eval": 0.3}

COVARIATES = (
    Covariate(name="x1"),
    Covariate(name="x2"),
    Covariate(name="x3"),
    Covariate(name="seg", kind="categorical"),
)

Cols = Mapping[str, np.ndarray]
Mask = Callable[[Cols], np.ndarray]

AUDIT_GROUPS: dict[str, Mask] = {
    "seg=a": lambda c: c["seg"] == "a",
    "seg=b": lambda c: c["seg"] == "b",
    "seg=c": lambda c: c["seg"] == "c",
    "x1>0": lambda c: c["x1"] > 0,
    "x2>0": lambda c: c["x2"] > 0,
    "x1>0&x2>0": lambda c: (c["x1"] > 0) & (c["x2"] > 0),
    "seg=c&x2>0.5": lambda c: (c["seg"] == "c") & (c["x2"] > 0.5),
    "seg=c&x1>1.5&x2>1": lambda c: (c["seg"] == "c") & (c["x1"] > 1.5) & (c["x2"] > 1.0),
}


def correction_cell(cols: Cols) -> np.ndarray:
    """Predeclared disjoint correction cells: segment x (x1 > 0) x (x2 > 0.5)."""
    seg = np.searchsorted(np.array(["a", "b", "c"]), cols["seg"])
    return seg * 4 + (cols["x1"] > 0) * 2 + (cols["x2"] > 0.5)


N_CELLS = 12


# Scenarios


def _mu0(c: Cols) -> np.ndarray:
    return 1.0 + np.sin(2.0 * c["x1"]) + 0.5 * c["x2"] ** 2 + 0.3 * (c["seg"] == "b")


def _tau_constant(c: Cols) -> np.ndarray:
    return np.full(c["x1"].shape, 0.25)


def _tau_linear(c: Cols) -> np.ndarray:
    return 0.10 + 0.30 * c["x1"] - 0.15 * c["x2"] + 0.20 * (c["seg"] == "b")


def _tau_nonlinear(c: Cols) -> np.ndarray:
    return -0.20 + 0.70 * ((c["x1"] > 0) & (c["x2"] > 0))


def _tau_subgroup(c: Cols) -> np.ndarray:
    small = (c["seg"] == "c") & (c["x2"] > 0.5)
    return np.where(small, -0.50, 0.15 + 0.25 * c["x1"])


@dataclass(frozen=True, slots=True)
class Scenario:
    """One synthetic design: effect surface, deployment cost and dependence."""

    name: str
    tau: Callable[[Cols], np.ndarray]
    cost: float
    role: str  # "constant", "linear", "nonlinear" or "subgroup"
    n_units: int = 20_000
    cluster_size: int = 1  # units per cluster; 1 means independent units
    repeats: int = 1  # rows per unit sharing x, d and a unit effect
    noise_sd: float = 1.0
    dependence_sd: float = 0.7


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario("constant", _tau_constant, cost=0.10, role="constant"),
        Scenario("linear", _tau_linear, cost=0.10, role="linear"),
        Scenario("nonlinear", _tau_nonlinear, cost=0.05, role="nonlinear"),
        Scenario("subgroup_miscalibration", _tau_subgroup, cost=0.10, role="subgroup"),
        Scenario(
            "nonlinear_clustered", _tau_nonlinear, cost=0.05, role="nonlinear", cluster_size=8
        ),
        Scenario(
            "nonlinear_repeated",
            _tau_nonlinear,
            cost=0.05,
            role="nonlinear",
            n_units=10_000,
            repeats=2,
        ),
    )
}


@dataclass(frozen=True, slots=True)
class Draw:
    """One simulated experiment, one row per observation."""

    cols: dict[str, np.ndarray]
    y: np.ndarray
    d: np.ndarray
    tau: np.ndarray
    unit_ids: np.ndarray
    cluster_ids: np.ndarray | None  # passed to splits and inference
    dependence: np.ndarray  # integer code of the independent sampling unit


def draw_covariates(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Pre-treatment covariates for ``n`` units.

    Parameters
    ----------
    n : int
        Number of units.
    rng : np.random.Generator
        Seeded generator.

    Returns
    -------
    dict[str, np.ndarray]
        Columns ``x1``, ``x2``, ``x3`` (standard normal; ``x3`` carries no
        signal) and ``seg`` in ``{"a", "b", "c"}``.
    """
    return {
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
        "seg": rng.choice(np.array(["a", "b", "c"]), size=n, p=[0.5, 0.3, 0.2]),
    }


def simulate(scenario: Scenario, rng: np.random.Generator) -> Draw:
    """Simulate one randomized experiment from *scenario*.

    Parameters
    ----------
    scenario : Scenario
        The design to draw from.
    rng : np.random.Generator
        Seeded generator.

    Returns
    -------
    Draw
        Rows, truth and identity columns.
    """
    n = scenario.n_units
    units = draw_covariates(n, rng)
    d_unit = rng.binomial(1, 0.5, size=n).astype(float)
    unit_effect = np.zeros(n)
    cluster_of_unit = np.arange(n) // scenario.cluster_size
    if scenario.cluster_size > 1:
        n_clusters = int(cluster_of_unit.max()) + 1
        unit_effect = rng.normal(scale=scenario.dependence_sd, size=n_clusters)[cluster_of_unit]
    if scenario.repeats > 1:
        unit_effect = unit_effect + rng.normal(scale=scenario.dependence_sd, size=n)
    row_unit = np.repeat(np.arange(n), scenario.repeats)
    cols = {k: v[row_unit] for k, v in units.items()}
    d = d_unit[row_unit]
    tau = scenario.tau(cols)
    y = (
        _mu0(cols)
        + d * tau
        + unit_effect[row_unit]
        + rng.normal(scale=scenario.noise_sd, size=row_unit.size)
    )
    unit_ids = np.array([f"u{i}" for i in row_unit])
    if scenario.cluster_size > 1:
        cluster_ids = np.array([f"c{k}" for k in cluster_of_unit[row_unit]])
        dependence = cluster_of_unit[row_unit]
    else:
        cluster_ids = None
        dependence = row_unit
    return Draw(cols, y, d, tau, unit_ids, cluster_ids, dependence)


def design_matrix(cols: Cols) -> np.ndarray:
    """Numeric matrix for the boosted learners: continuous columns plus segment dummies.

    Parameters
    ----------
    cols : Cols
        Covariate columns.

    Returns
    -------
    np.ndarray
        ``[x1, x2, x3, seg==b, seg==c]``.
    """
    return np.column_stack(
        [cols["x1"], cols["x2"], cols["x3"], cols["seg"] == "b", cols["seg"] == "c"]
    ).astype(float)


def _take(cols: Cols, mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in cols.items()}


# Splits and leakage checks


def three_way_split(draw: Draw, seed: int) -> np.ndarray:
    """Label every row 0 (fit), 1 (selection) or 2 (evaluation).

    Units, or whole clusters when the draw has them, are kept on one side; a
    repeated unit id is its own singleton cluster. Stratified on treatment.

    Parameters
    ----------
    draw : Draw
        The experiment to split.
    seed : int
        Split seed.

    Returns
    -------
    np.ndarray
        Integer split label per row.
    """
    labels = np.zeros(draw.y.size, dtype=np.int64)
    evaluation = outer_split(
        draw.unit_ids,
        test_size=SPLIT_SHARES["eval"],
        seed=seed,
        stratify=draw.d,
        cluster_ids=draw.cluster_ids,
    )
    rest = ~evaluation
    select_share = SPLIT_SHARES["select"] / (SPLIT_SHARES["fit"] + SPLIT_SHARES["select"])
    selection = outer_split(
        draw.unit_ids[rest],
        test_size=select_share,
        seed=seed + 1,
        stratify=draw.d[rest],
        cluster_ids=None if draw.cluster_ids is None else draw.cluster_ids[rest],
    )
    labels[evaluation] = 2
    labels[np.flatnonzero(rest)[selection]] = 1
    assert_no_leakage(draw, labels)
    return labels


def assert_no_leakage(draw: Draw, labels: np.ndarray) -> None:
    """Refuse a split that separates a unit's rows or a cluster's members.

    Parameters
    ----------
    draw : Draw
        The split experiment.
    labels : np.ndarray
        Split label per row.
    """
    check_cluster_atomic(draw.unit_ids, labels, what="unit_ids")
    if draw.cluster_ids is not None:
        check_cluster_atomic(draw.cluster_ids, labels, what="cluster_ids")


# Held-out estimation primitives


def heldout_score(y: np.ndarray, d: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Per-row score with conditional mean ``tau(x)`` under randomization.

    Parameters
    ----------
    y : np.ndarray
        Outcomes on the scored split.
    d : np.ndarray
        Treatment on the scored split.
    m : np.ndarray
        Outcome-mean predictions from a model fitted on another split.

    Returns
    -------
    np.ndarray
        ``(y - m) (d - p) / (p (1 - p))`` with ``p`` the split's treated share.
    """
    p = float(d.mean())
    return (y - m) * (d - p) / (p * (1.0 - p))


def joint_mean_cov(
    contributions: np.ndarray, dependence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Column means and their dependence-robust covariance.

    Parameters
    ----------
    contributions : np.ndarray
        ``(n, k)`` influence-scale contributions; column means are reported.
    dependence : np.ndarray
        Integer code of each row's independent sampling unit.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Means and ``k x k`` covariance with the ``K / (K - 1)`` correction.
    """
    u = np.atleast_2d(contributions.T).T
    n = u.shape[0]
    centered = u - u.mean(axis=0)
    codes = np.unique(dependence, return_inverse=True)[1]
    k = int(codes.max()) + 1
    totals = np.zeros((k, u.shape[1]))
    np.add.at(totals, codes, centered)
    cov = (k / (k - 1)) * totals.T @ totals / n**2
    return u.mean(axis=0), cov


@dataclass(frozen=True, slots=True)
class Interval:
    """A held-out estimate and its normal interval."""

    estimate: float
    se: float
    lower: float
    upper: float


def _interval(estimate: float, se: float, crit: float) -> Interval:
    return Interval(estimate, se, estimate - crit * se, estimate + crit * se)


def max_abs_critical(corr: np.ndarray, alpha: float, seed: int = 20260212) -> float:
    """Two-sided simultaneous critical value of a Gaussian vector with correlation ``corr``.

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix (may be singular).
    alpha : float
        Familywise error.
    seed : int
        Seed of the fixed Monte Carlo draw.

    Returns
    -------
    float
        The ``1 - alpha`` quantile of ``max_g |Z_g|``.
    """
    if corr.shape[0] == 1:
        return float(norm.isf(alpha / 2))
    vals, vecs = np.linalg.eigh(corr)
    root = vecs * np.sqrt(np.clip(vals, 0.0, None))
    z = np.random.default_rng(seed).standard_normal((200_000, corr.shape[0])) @ root.T
    return float(np.quantile(np.abs(z).max(axis=1), 1.0 - alpha))


# Subgroup audit


@dataclass(frozen=True, slots=True)
class GroupRow:
    """One predeclared group: predicted versus experimental effect, or why not."""

    group: str
    n_treated: int
    n_control: int
    clusters_treated: int
    clusters_control: int
    predicted: float | None
    experimental: Interval | None
    discrepancy: Interval | None
    unavailable_reason: str | None
    true_discrepancy: float | None = None


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Joint audit of predicted against randomized subgroup effects."""

    rows: tuple[GroupRow, ...]
    critical_value: float
    wald_statistic: float | None
    wald_df: int
    wald_p_value: float | None

    @property
    def auditable(self) -> tuple[GroupRow, ...]:
        """Rows inside the joint family."""
        return tuple(r for r in self.rows if r.discrepancy is not None)


def audit_groups(
    tau_hat: np.ndarray,
    psi: np.ndarray,
    d: np.ndarray,
    cols: Cols,
    dependence: np.ndarray,
    *,
    groups: Mapping[str, Mask] = AUDIT_GROUPS,
    alpha: float = ALPHA,
) -> AuditReport:
    """Compare group means of predictions with randomized group effects.

    ``delta_g = mean_g(tau_hat) - mean_g(psi)`` is estimated with the
    ratio-of-means influence ``1{i in g}(tau_hat_i - psi_i - delta_g) / p_g``;
    overlapping groups share rows and therefore covariance.

    Parameters
    ----------
    tau_hat : np.ndarray
        Fixed predictions on the evaluation rows.
    psi : np.ndarray
        Held-out scores on the same rows.
    d : np.ndarray
        Treatment on the same rows.
    cols : Cols
        Covariates on the same rows.
    dependence : np.ndarray
        Independent sampling unit code per row.
    groups : Mapping[str, Mask]
        Predeclared, possibly overlapping groups.
    alpha : float
        Familywise error of the simultaneous intervals and the Wald test.

    Returns
    -------
    AuditReport
        One row per group; groups lacking evidence are unavailable rows.
    """
    names, masks, counts = [], [], []
    for name, rule in groups.items():
        mask = np.asarray(rule(cols), dtype=bool)
        treated, control = mask & (d == 1), mask & (d == 0)
        count = (
            int(treated.sum()),
            int(control.sum()),
            np.unique(dependence[treated]).size,
            np.unique(dependence[control]).size,
        )
        names.append(name)
        masks.append(mask)
        counts.append(count)
    ok = [min(c[0], c[1]) >= MIN_ARM_UNITS and min(c[2], c[3]) >= MIN_ARM_CLUSTERS for c in counts]
    kept = [i for i, good in enumerate(ok) if good]
    columns = []
    for i in kept:
        mask = masks[i]
        share = mask.mean()
        r = tau_hat - psi
        delta = r[mask].mean()
        effect = psi[mask].mean()
        columns.append(mask * (r - delta) / share + delta)
        columns.append(mask * (psi - effect) / share + effect)
    rows: list[GroupRow] = []
    crit = math.nan
    wald = p_value = None
    df = 0
    if kept:
        means, cov = joint_mean_cov(np.column_stack(columns), dependence)
        disc_idx = np.arange(0, 2 * len(kept), 2)
        disc_cov = cov[np.ix_(disc_idx, disc_idx)]
        se = np.sqrt(np.diag(disc_cov))
        crit = max_abs_critical(disc_cov / np.outer(se, se), alpha)
        df = int(np.linalg.matrix_rank(disc_cov))
        wald = float(means[disc_idx] @ np.linalg.pinv(disc_cov) @ means[disc_idx])
        p_value = float(chi2.sf(wald, df))
        z_point = float(norm.isf(alpha / 2))
        joint = dict(zip(kept, range(len(kept)), strict=True))
    for i, name in enumerate(names):
        c = counts[i]
        sizes = {
            "n_treated": c[0],
            "n_control": c[1],
            "clusters_treated": c[2],
            "clusters_control": c[3],
        }
        if i not in kept:
            reason = (
                f"insufficient_evidence: arms have {c[0]}/{c[1]} rows and {c[2]}/{c[3]} "
                f"independent units; need {MIN_ARM_UNITS} rows and {MIN_ARM_CLUSTERS} per arm"
            )
            rows.append(
                GroupRow(
                    name,
                    **sizes,
                    predicted=None,
                    experimental=None,
                    discrepancy=None,
                    unavailable_reason=reason,
                )
            )
            continue
        j = joint[i]
        delta, effect = means[2 * j], means[2 * j + 1]
        rows.append(
            GroupRow(
                name,
                **sizes,
                predicted=float(tau_hat[masks[i]].mean()),
                experimental=_interval(effect, math.sqrt(cov[2 * j + 1, 2 * j + 1]), z_point),
                discrepancy=_interval(delta, se[j], crit),
                unavailable_reason=None,
            )
        )
    return AuditReport(tuple(rows), crit, wald, df, p_value)


# Group-bias corrections


@dataclass(frozen=True, slots=True)
class CellCorrection:
    """Per-cell offsets subtracted from predictions; fitted on the selection split."""

    offsets: np.ndarray  # length N_CELLS, zero for cells without evidence
    shrinkage: np.ndarray

    def apply(self, tau_hat: np.ndarray, cols: Cols) -> np.ndarray:
        """Corrected predictions.

        Parameters
        ----------
        tau_hat : np.ndarray
            Uncorrected predictions.
        cols : Cols
            Covariates of the same rows.

        Returns
        -------
        np.ndarray
            ``tau_hat`` minus the offset of each row's cell.
        """
        return tau_hat - self.offsets[correction_cell(cols)]


def fit_cell_correction(
    tau_hat: np.ndarray,
    psi: np.ndarray,
    d: np.ndarray,
    cols: Cols,
    dependence: np.ndarray,
    *,
    shrink: bool,
) -> CellCorrection:
    """Estimate per-cell group bias on selection rows and optionally shrink it.

    The naive correction subtracts each cell's estimated bias ``b`` in full,
    which removes bias only in expectation. The shrunk correction subtracts
    ``lambda b`` with ``lambda = max(0, 1 - v / b^2)``, the positive-part
    plug-in for the MSE-optimal ``b_true^2 / (b_true^2 + v)``. Cells below the
    audit evidence floor keep a zero offset.

    Parameters
    ----------
    tau_hat : np.ndarray
        Predictions on selection rows.
    psi : np.ndarray
        Held-out scores on selection rows.
    d : np.ndarray
        Treatment on selection rows.
    cols : Cols
        Covariates on selection rows.
    dependence : np.ndarray
        Independent sampling unit per selection row.
    shrink : bool
        Whether to shrink offsets.

    Returns
    -------
    CellCorrection
        Offsets to subtract per cell.
    """
    groups = {str(k): (lambda c, k=k: correction_cell(c) == k) for k in range(N_CELLS)}
    report = audit_groups(tau_hat, psi, d, cols, dependence, groups=groups)
    offsets = np.zeros(N_CELLS)
    lam = np.zeros(N_CELLS)
    for row in report.rows:
        if row.discrepancy is None:
            continue
        k = int(row.group)
        b, v = row.discrepancy.estimate, row.discrepancy.se**2
        lam[k] = max(0.0, 1.0 - v / b**2) if shrink and b != 0.0 else float(not shrink)
        offsets[k] = lam[k] * b
    return CellCorrection(offsets, lam)


# One replicate


Predictor = Callable[[Cols], np.ndarray]


@dataclass
class Fitted:
    """Learners and nuisance fitted without evaluation rows."""

    predictors: dict[str, Predictor]
    simple_treat_all: bool
    rl_rounds: int | None
    unavailable: dict[str, str] = field(default_factory=dict)


def fit_learners(
    draw: Draw, labels: np.ndarray, seed: int, scenario: Scenario
) -> tuple[Fitted, BoostedRegressor]:
    """Fit both learners, the control variate, the simple policy and corrections.

    Parameters
    ----------
    draw : Draw
        The experiment.
    labels : np.ndarray
        Split labels from :func:`three_way_split`.
    seed : int
        Seed for the R-learner's cross-fitting folds.
    scenario : Scenario
        Supplies the deployment cost.

    Returns
    -------
    tuple[Fitted, BoostedRegressor]
        Fitted predictors and the evaluation control-variate model.
    """
    fit, sel = labels == 0, labels == 1
    cols_fit, cols_sel = _take(draw.cols, fit), _take(draw.cols, sel)
    X = design_matrix(draw.cols)
    control_variate = BoostedRegressor().fit(X[fit], draw.y[fit])
    predictors: dict[str, Predictor] = {}
    unavailable: dict[str, str] = {}
    try:
        cate = fit_cate(
            draw.y[fit],
            draw.d[fit],
            cols_fit,
            interact=COVARIATES,
            cluster_ids=_dependence_ids(draw, fit),
        )
        predictors["current"] = lambda c, cate=cate: np.asarray(cate.score(c), dtype=float)
    except CodedError as error:
        unavailable["current"] = f"{error.code}"
    rl = RLearner(seed=seed).fit(
        X[fit], draw.y[fit], draw.d[fit], draw.unit_ids[fit], _cluster_ids(draw, fit)
    )
    rl.select(X[sel], draw.y[sel], draw.d[sel])
    predictors["candidate"] = lambda c, rl=rl: rl.predict(design_matrix(c))
    psi_sel = heldout_score(draw.y[sel], draw.d[sel], control_variate.predict(X[sel]))
    simple_treat_all = bool(psi_sel.mean() > scenario.cost)
    for base in list(predictors):
        tau_sel = predictors[base](cols_sel)
        for shrink, tag in ((True, "shrunk"), (False, "naive")):
            correction = fit_cell_correction(
                tau_sel, psi_sel, draw.d[sel], cols_sel, draw.dependence[sel], shrink=shrink
            )
            predictors[f"{base}+{tag}"] = lambda c, b=predictors[base], k=correction: k.apply(
                b(c), c
            )
    return Fitted(predictors, simple_treat_all, rl.n_rounds, unavailable), control_variate


def _cluster_ids(draw: Draw, mask: np.ndarray) -> np.ndarray | None:
    return None if draw.cluster_ids is None else draw.cluster_ids[mask]


def _dependence_ids(draw: Draw, mask: np.ndarray) -> np.ndarray | None:
    if draw.cluster_ids is not None:
        return draw.cluster_ids[mask]
    if np.unique(draw.unit_ids[mask]).size < mask.sum():
        return draw.unit_ids[mask]
    return None


@dataclass
class Reference:
    """A large independent covariate draw for population truths."""

    cols: dict[str, np.ndarray]
    tau: np.ndarray

    @classmethod
    def build(cls, scenario: Scenario, seed: int, n: int = REFERENCE_N) -> Reference:
        """Draw the reference population for *scenario*.

        Parameters
        ----------
        scenario : Scenario
            Supplies the effect surface.
        seed : int
            Reference seed.
        n : int
            Reference size.

        Returns
        -------
        Reference
            Covariates and true effects.
        """
        cols = draw_covariates(n, np.random.default_rng(seed))
        return cls(cols, scenario.tau(cols))


def evaluate(
    draw: Draw,
    labels: np.ndarray,
    fitted: Fitted,
    control_variate: BoostedRegressor,
    scenario: Scenario,
    reference: Reference,
) -> dict:
    """Every reported number, from evaluation rows only, plus population truths.

    Parameters
    ----------
    draw : Draw
        The experiment.
    labels : np.ndarray
        Split labels.
    fitted : Fitted
        Output of :func:`fit_learners`.
    control_variate : BoostedRegressor
        Outcome model fitted on the fit split.
    scenario : Scenario
        Supplies the cost.
    reference : Reference
        Population truth.

    Returns
    -------
    dict
        Policy values, primary comparisons, audits and correction effects.
    """
    ev = labels == 2
    cols = _take(draw.cols, ev)
    d, dep = draw.d[ev], draw.dependence[ev]
    psi = heldout_score(draw.y[ev], d, control_variate.predict(design_matrix(cols)))
    cost = scenario.cost
    tau_hat = {k: f(cols) for k, f in fitted.predictors.items()}
    tau_ref = {k: f(reference.cols) for k, f in fitted.predictors.items()}
    n_ev, n_ref = psi.size, reference.tau.size
    decisions = {k: (t > cost).astype(float) for k, t in tau_hat.items()}
    decisions_ref = {k: (t > cost).astype(float) for k, t in tau_ref.items()}
    decisions["treat_all"], decisions_ref["treat_all"] = np.ones(n_ev), np.ones(n_ref)
    decisions["treat_none"], decisions_ref["treat_none"] = np.zeros(n_ev), np.zeros(n_ref)
    simple = "treat_all" if fitted.simple_treat_all else "treat_none"
    decisions["simple"], decisions_ref["simple"] = decisions[simple], decisions_ref[simple]
    decisions_ref["oracle"] = (reference.tau > cost).astype(float)
    z = float(norm.isf(ALPHA / 2))
    names = list(decisions)
    means, cov = joint_mean_cov(np.column_stack([decisions[k] * (psi - cost) for k in names]), dep)
    truth = {k: float(np.mean(decisions_ref[k] * (reference.tau - cost))) for k in decisions_ref}
    values = {
        k: {
            **asdict(_interval(float(means[i]), math.sqrt(cov[i, i]), z)),
            "true": truth[k],
            "treated_share": float(decisions[k].mean()),
        }
        for i, k in enumerate(names)
    }

    def compare(a: str, b: str) -> dict:
        i, j = names.index(a), names.index(b)
        se = math.sqrt(max(cov[i, i] + cov[j, j] - 2 * cov[i, j], 0.0))
        return {
            **asdict(_interval(float(means[i] - means[j]), se, z)),
            "true": truth[a] - truth[b],
            "decision_disagreement": float(np.mean(decisions[a] != decisions[b])),
        }

    comparisons: dict[str, dict] = {}
    if "current" in tau_hat:
        comparisons["candidate-current"] = compare("candidate", "current")
    comparisons["candidate-simple"] = compare("candidate", "simple")
    if "current" in tau_hat:
        comparisons["current-simple"] = compare("current", "simple")
    for k in tau_hat:
        if "+" in k:
            comparisons[f"{k}-{k.split('+')[0]}"] = compare(k, k.split("+")[0])

    audits = {}
    for k, t in tau_hat.items():
        report = audit_groups(t, psi, d, cols, dep)
        rows = []
        for row in report.rows:
            mask = AUDIT_GROUPS[row.group](reference.cols)
            true = float(np.mean(tau_ref[k][mask] - reference.tau[mask]))
            rows.append(asdict(row) | {"true_discrepancy": true})
        audits[k] = {
            "rows": rows,
            "critical_value": report.critical_value,
            "wald_statistic": report.wald_statistic,
            "wald_df": report.wald_df,
            "wald_p_value": report.wald_p_value,
            "rmse": float(np.sqrt(np.mean((tau_ref[k] - reference.tau) ** 2))),
            "true_mean_abs_discrepancy": float(np.mean([abs(r["true_discrepancy"]) for r in rows])),
        }
    return {
        "simple_policy": simple,
        "rl_rounds": fitted.rl_rounds,
        "unavailable": fitted.unavailable,
        "values": values,
        "oracle_value": truth["oracle"],
        "comparisons": comparisons,
        "audits": audits,
        "tau_sd": float(reference.tau.std()),
    }


def run_replicate(scenario: Scenario, seed: int, reference: Reference) -> dict:
    """Simulate, split, fit and evaluate once.

    Parameters
    ----------
    scenario : Scenario
        Design.
    seed : int
        Replicate seed.
    reference : Reference
        Population truth for *scenario*.

    Returns
    -------
    dict
        The replicate's evaluation record.
    """
    draw = simulate(scenario, np.random.default_rng(seed))
    labels = three_way_split(draw, seed)
    fitted, control_variate = fit_learners(draw, labels, seed, scenario)
    return evaluate(draw, labels, fitted, control_variate, scenario, reference)


# Isolated audit calibration


def audit_coverage(name: str, reps: int, seed: int = 1000) -> dict:
    """Simultaneous coverage of the subgroup audit with nothing refitted.

    Predictions are a fixed, deliberately imperfect function
    ``0.1 + 0.25 x1`` and the control variate is fitted once on an
    independent draw, so each replicate only redraws an evaluation-sized split.
    This isolates the audit's own calibration from learner variability and
    costs milliseconds per replicate.

    Parameters
    ----------
    name : str
        Scenario supplying the design.
    reps : int
        Replicates.
    seed : int
        Base seed; replicate ``r`` of the scenario at position ``s`` uses
        ``seed + 100_000 s + r``, so designs never share draws.

    Returns
    -------
    dict
        Simultaneous coverage with its binomial standard error, and per-group
        mean and standard deviation of the discrepancy z-scores.
    """
    base = SCENARIOS[name]
    seed += 100_000 * list(SCENARIOS).index(name)
    n_eval = round(base.n_units * SPLIT_SHARES["eval"])
    scenario = replace(base, n_units=n_eval)
    reference = Reference.build(scenario, seed=seed - 1)
    train = simulate(base, np.random.default_rng(seed - 2))
    control_variate = BoostedRegressor().fit(design_matrix(train.cols), train.y)

    def fixed(c: Cols) -> np.ndarray:
        return 0.1 + 0.25 * c["x1"]

    truth = {}
    for g, rule in AUDIT_GROUPS.items():
        mask = rule(reference.cols)
        truth[g] = float(np.mean(fixed(reference.cols)[mask] - reference.tau[mask]))
    covered = []
    z: dict[str, list[float]] = {g: [] for g in AUDIT_GROUPS}
    for r in range(reps):
        draw = simulate(scenario, np.random.default_rng(seed + r))
        psi = heldout_score(draw.y, draw.d, control_variate.predict(design_matrix(draw.cols)))
        report = audit_groups(fixed(draw.cols), psi, draw.d, draw.cols, draw.dependence)
        hit = True
        for row in report.auditable:
            assert row.discrepancy is not None
            interval, t = row.discrepancy, truth[row.group]
            hit = hit and interval.lower <= t <= interval.upper
            z[row.group].append((interval.estimate - t) / interval.se)
        covered.append(hit)
    rate = float(np.mean(covered))
    return {
        "scenario": name,
        "replicates": reps,
        "simultaneous_coverage": rate,
        "coverage_se": math.sqrt(rate * (1 - rate) / reps),
        "z": {
            g: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
            for g, v in z.items()
            if len(v) > 1
        },
    }


# Decision stability (representative case)


def decision_stability(
    scenario: Scenario, seed: int, reference: Reference, n_boot: int = 5
) -> dict:
    """Refit on resampled fit splits and measure held-out decision flips.

    Whole clusters (or units) of the fit split are resampled with replacement;
    duplicated copies keep their ids, so cross-fitting keeps them in one fold.
    Selection and evaluation rows are untouched.

    Parameters
    ----------
    scenario : Scenario
        Design.
    seed : int
        Replicate seed (same draw and split as :func:`run_replicate`).
    reference : Reference
        Population truth.
    n_boot : int
        Number of resampled refits.

    Returns
    -------
    dict
        Per learner: mean and maximum share of evaluation units whose decision
        flips, and the spread of the true policy value across refits.
    """
    draw = simulate(scenario, np.random.default_rng(seed))
    labels = three_way_split(draw, seed)
    base, _ = fit_learners(draw, labels, seed, scenario)
    ev_cols = _take(draw.cols, labels == 2)
    base_dec = {k: base.predictors[k](ev_cols) > scenario.cost for k in ("current", "candidate")}
    rng = np.random.default_rng(seed + 7)
    fit_rows = np.flatnonzero(labels == 0)
    groups = draw.dependence[fit_rows]
    unique = np.unique(groups)
    flips: dict[str, list[float]] = {k: [] for k in base_dec}
    values: dict[str, list[float]] = {k: [] for k in base_dec}
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([fit_rows[groups == g] for g in chosen])
        rows = np.concatenate([rows, np.flatnonzero(labels != 0)])
        resampled = Draw(
            {k: v[rows] for k, v in draw.cols.items()},
            draw.y[rows],
            draw.d[rows],
            draw.tau[rows],
            draw.unit_ids[rows],
            None if draw.cluster_ids is None else draw.cluster_ids[rows],
            draw.dependence[rows],
        )
        refit, _ = fit_learners(resampled, labels[rows], seed, scenario)
        for k in base_dec:
            flips[k].append(
                float(np.mean((refit.predictors[k](ev_cols) > scenario.cost) != base_dec[k]))
            )
            pol = refit.predictors[k](reference.cols) > scenario.cost
            values[k].append(float(np.mean(pol * (reference.tau - scenario.cost))))
    return {
        k: {
            "mean_flip_share": float(np.mean(flips[k])),
            "max_flip_share": float(np.max(flips[k])),
            "true_value_sd": float(np.std(values[k], ddof=1)) if n_boot > 1 else None,
        }
        for k in base_dec
    }


# Campaign summary and recommendation


def _covered(row: dict) -> bool:
    disc = row["discrepancy"]
    return disc["lower"] <= row["true_discrepancy"] <= disc["upper"]


def summarize(scenario: Scenario, records: list[dict]) -> dict:
    """Monte Carlo summary of one scenario's replicates.

    Parameters
    ----------
    scenario : Scenario
        Design.
    records : list[dict]
        Replicate records.

    Returns
    -------
    dict
        Rates and means used by :func:`recommend`.
    """
    reps = len(records)
    out: dict = {"scenario": scenario.name, "role": scenario.role, "replicates": reps}
    oracle = float(np.mean([r["oracle_value"] for r in records]))
    out["true_value"] = {
        k: float(np.mean([r["values"][k]["true"] for r in records])) for k in records[0]["values"]
    }
    out["oracle_value"] = oracle
    out["value_ci_coverage"] = {
        k: float(
            np.mean(
                [v["lower"] <= v["true"] <= v["upper"] for v in (r["values"][k] for r in records)]
            )
        )
        for k in records[0]["values"]
    }
    comp = {}
    for name in records[0]["comparisons"]:
        rows = [r["comparisons"][name] for r in records]
        comp[name] = {
            "mean_true": float(np.mean([c["true"] for c in rows])),
            "mean_estimate": float(np.mean([c["estimate"] for c in rows])),
            "claims_gain": float(np.mean([c["lower"] > 0 for c in rows])),
            "claims_loss": float(np.mean([c["upper"] < 0 for c in rows])),
            "false_gain_claim": float(np.mean([c["lower"] > 0 and c["true"] <= 0 for c in rows])),
            "coverage": float(np.mean([c["lower"] <= c["true"] <= c["upper"] for c in rows])),
            "decision_disagreement": float(np.mean([c["decision_disagreement"] for c in rows])),
        }
    out["comparisons"] = comp
    both = [
        r["comparisons"]["candidate-simple"]["lower"] > 0
        and r["comparisons"].get("candidate-current", {"lower": -1.0})["lower"] > 0
        for r in records
    ]
    out["candidate_primary_win_rate"] = float(np.mean(both))
    audits = {}
    for k in records[0]["audits"]:
        per = [r["audits"][k] for r in records]
        simultaneous = [all(_covered(row) for row in a["rows"] if row["discrepancy"]) for a in per]
        groups = {}
        for g in AUDIT_GROUPS:
            rows = [next(row for row in a["rows"] if row["group"] == g) for a in per]
            avail = [row for row in rows if row["discrepancy"]]
            groups[g] = {
                "available_rate": len(avail) / reps,
                "coverage": float(np.mean([_covered(row) for row in avail])) if avail else None,
                "mean_true_discrepancy": float(np.mean([row["true_discrepancy"] for row in rows])),
                "detected_rate": float(
                    np.mean(
                        [
                            row["discrepancy"]["lower"] > 0 or row["discrepancy"]["upper"] < 0
                            for row in avail
                        ]
                    )
                )
                if avail
                else None,
            }
        audits[k] = {
            "simultaneous_coverage": float(np.mean(simultaneous)),
            "wald_rejection_rate": float(
                np.mean([a["wald_p_value"] is not None and a["wald_p_value"] < ALPHA for a in per])
            ),
            "rmse": float(np.mean([a["rmse"] for a in per])),
            "true_mean_abs_discrepancy": float(
                np.mean([a["true_mean_abs_discrepancy"] for a in per])
            ),
            "true_value": out["true_value"].get(k),
            "groups": groups,
        }
    out["audits"] = audits
    out["tau_sd"] = float(np.mean([r["tau_sd"] for r in records]))
    out["unavailable"] = sorted({v for r in records for v in r["unavailable"].values()})
    return out


def recommend(summaries: list[dict]) -> dict:
    """Apply the predeclared implement/defer/reject rule.

    ``reject`` when the candidate's mean true value gain over the current
    learner does not close ``MATERIAL_GAP_SHARE`` of the current-to-oracle gap
    in every nonlinear scenario. Otherwise ``implement`` only when every safety
    check passes, else ``defer`` naming the failures. ``implement`` approves
    design work; production use still requires the method acceptance
    requirements in ``CONTRIBUTING.md``.

    Parameters
    ----------
    summaries : list[dict]
        Output of :func:`summarize` per scenario.

    Returns
    -------
    dict
        Verdict, per-criterion evidence, and failures.
    """
    criteria: dict[str, dict] = {}
    failures: list[str] = []
    gains = []
    for s in summaries:
        tv = s["true_value"]
        name = s["scenario"]
        if s["role"] == "nonlinear":
            gap = s["oracle_value"] - tv["current"]
            share = (tv["candidate"] - tv["current"]) / gap if gap > 0 else 0.0
            gains.append(share)
            criteria[f"{name}: gap closed"] = {"value": share, "threshold": MATERIAL_GAP_SHARE}
            win = s["candidate_primary_win_rate"]
            criteria[f"{name}: primary win rate"] = {"value": win, "threshold": None}
        if s["role"] in ("linear", "constant", "subgroup"):
            loss = (tv["candidate"] - tv["current"]) / max(s["oracle_value"], 1e-12)
            criteria[f"{name}: value vs current / oracle"] = {
                "value": loss,
                "threshold": -HARM_SHARE,
            }
            if loss < -HARM_SHARE:
                failures.append(f"{name}: candidate loses {-loss:.1%} of oracle value to current")
        if s["role"] == "constant":
            for k in ("candidate-simple", "current-simple"):
                false = s["comparisons"][k]["false_gain_claim"]
                criteria[f"{name}: false targeting claim {k}"] = {
                    "value": false,
                    "threshold": ALPHA + MC_TOLERANCE,
                }
                if k.startswith("candidate") and false > ALPHA + MC_TOLERANCE:
                    failures.append(
                        f"{name}: candidate claims a false targeting gain at {false:.1%}"
                    )
        for k in ("current", "candidate"):
            if k not in s["audits"]:
                continue
            cov = s["audits"][k]["simultaneous_coverage"]
            criteria[f"{name}: audit coverage {k}"] = {
                "value": cov,
                "threshold": 1 - ALPHA - MC_TOLERANCE,
            }
            if cov < 1 - ALPHA - MC_TOLERANCE:
                failures.append(f"{name}: {k} audit simultaneous coverage {cov:.3f}")
        cov = s["comparisons"]["candidate-simple"]["coverage"]
        criteria[f"{name}: candidate-simple interval coverage"] = {
            "value": cov,
            "threshold": 1 - ALPHA - MC_TOLERANCE,
        }
        if cov < 1 - ALPHA - MC_TOLERANCE:
            failures.append(f"{name}: candidate-simple value interval coverage {cov:.3f}")
    if not gains or min(gains) < MATERIAL_GAP_SHARE:
        verdict = "reject"
    elif failures:
        verdict = "defer"
    else:
        verdict = "implement"
    return {"verdict": verdict, "failures": failures, "criteria": criteria}


def _replicate_seed(seed: int, s_index: int, r: int) -> int:
    return seed + 1000 * s_index + r


def run_scenario(
    name: str,
    reps: int,
    seed: int,
    log: Callable[[str], None],
    journal: Path | None = None,
) -> list[dict]:
    """Run one scenario, resuming from and appending to an optional JSONL journal.

    Replicate ``r`` of the scenario at position ``s`` in :data:`SCENARIOS`
    uses seed ``seed + 1000 s + r``, so a resumed or sharded run reproduces an
    uninterrupted one.

    Parameters
    ----------
    name : str
        Scenario name.
    reps : int
        Replicates required.
    seed : int
        Base seed.
    log : Callable[[str], None]
        Progress sink.
    journal : Path | None
        File holding one replicate record per line; existing lines are reused.

    Returns
    -------
    list[dict]
        The first *reps* replicate records.
    """
    scenario = SCENARIOS[name]
    s_index = list(SCENARIOS).index(name)
    records: list[dict] = []
    if journal is not None and journal.exists():
        records = [json.loads(line) for line in journal.read_text().splitlines() if line]
    for record, r in zip(records, range(reps), strict=False):
        if record["seed"] != _replicate_seed(seed, s_index, r):
            raise ValueError(f"{journal} was written with a different base seed")
    if len(records) >= reps:
        return records[:reps]
    reference = Reference.build(scenario, seed=seed + 999_999)
    start = time.perf_counter()
    for r in range(len(records), reps):
        replicate_seed = _replicate_seed(seed, s_index, r)
        record = run_replicate(scenario, replicate_seed, reference) | {"seed": replicate_seed}
        records.append(record)
        if journal is not None:
            with journal.open("a") as handle:
                handle.write(json.dumps(record, default=float) + "\n")
        log(f"{name}: replicate {r + 1}/{reps} ({time.perf_counter() - start:.0f}s)")
    return records


def run_campaign(
    names: list[str],
    reps: int,
    seed: int,
    log: Callable[[str], None],
    journal_dir: Path | None = None,
) -> dict:
    """Run every named scenario for *reps* replicates and apply the rule.

    Parameters
    ----------
    names : list[str]
        Scenario names.
    reps : int
        Replicates per scenario.
    seed : int
        Base seed (see :func:`run_scenario`).
    log : Callable[[str], None]
        Progress sink.
    journal_dir : Path | None
        Directory of per-scenario JSONL journals for resumable runs.

    Returns
    -------
    dict
        Summaries and the recommendation.
    """
    summaries = []
    for name in names:
        journal = None if journal_dir is None else journal_dir / f"{name}.jsonl"
        records = run_scenario(name, reps, seed, log, journal)
        summaries.append(summarize(SCENARIOS[name], records))
    return {"summaries": summaries, "recommendation": recommend(summaries)}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv : list[str] | None
        Arguments; defaults to ``sys.argv``.

    Returns
    -------
    int
        Exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--case", default="nonlinear_clustered", help="representative use case")
    parser.add_argument("--stability-refits", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("nonlinear_targeting.json"))
    parser.add_argument("--journal-dir", type=Path, default=None)
    parser.add_argument(
        "--only-replicates", action="store_true", help="fill journals and stop (for sharding)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="extend an existing journal directory"
    )
    args = parser.parse_args(argv)
    if args.journal_dir is not None:
        if args.journal_dir.exists() and any(args.journal_dir.iterdir()) and not args.resume:
            parser.error(f"{args.journal_dir} is not empty; pass --resume to extend it")
        args.journal_dir.mkdir(parents=True, exist_ok=True)
    if args.only_replicates:
        for name in args.scenarios:
            journal = None if args.journal_dir is None else args.journal_dir / f"{name}.jsonl"
            run_scenario(name, args.reps, args.seed, print, journal)
        return 0
    result = run_campaign(args.scenarios, args.reps, args.seed, print, args.journal_dir)
    case = SCENARIOS[args.case]
    reference = Reference.build(case, seed=args.seed + 999_999)
    result["representative_case"] = {
        "scenario": case.name,
        "record": run_replicate(case, args.seed, reference),
        "stability": decision_stability(case, args.seed, reference, args.stability_refits),
    }
    args.out.write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result["recommendation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
