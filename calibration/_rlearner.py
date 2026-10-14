"""Research candidate for issue #73: an R-learner with a boosted base regressor.

This module is evidence-harness code, not a shipped estimator. It lives under
``calibration/`` so nothing here becomes public API before the method
acceptance requirements in ``CONTRIBUTING.md`` are met.

The R-learner (Nie and Wager 2021) minimizes the Robinson residual loss

    sum_i (d_i - e)^2 * ((y_i - m(x_i)) / (d_i - e) - tau(x_i))^2

with ``m`` the cross-fitted outcome mean and ``e`` the randomized treated
share. Under randomized assignment with a known constant ``e``, the population
minimizer is ``tau(x)`` for any ``m`` that depends on ``x`` only, so ``m``
moves variance, not identification. Cross-fitting reuses
:func:`increment.estimation.crossfit.fold_assignments`, so repeated unit ids
and clusters stay inside one fold.

No interval is attached to an R-learner prediction: the linear sandwich of the
interacted regression does not transfer, and every reported uncertainty in the
study comes from untouched held-out data instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from increment.estimation.crossfit import fold_assignments

# Fixed from a pilot on seeds 11 and 12, disjoint from every campaign seed.
# Zero rounds is the constant (ATE) model, so the learner can decline to
# find heterogeneity.
ROUNDS_GRID = (0, 10, 25, 50, 100, 200, 400)


@dataclass(frozen=True, slots=True)
class _Tree:
    feature: np.ndarray  # -1 marks a leaf
    split_bin: np.ndarray  # rows with bin <= split_bin go left
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray


class BoostedRegressor:
    """Weighted least-squares gradient boosting over histogram-binned trees.

    Columns are binned once at quantile edges learned on the fitting rows, and
    each tree is grown depth-first to ``max_depth`` using the exact weighted
    squared-error gain within those bins. ``min_leaf`` and ``l2`` are expressed
    in units of the mean row weight, so their meaning does not depend on the
    scale of the R-loss weights. Deterministic given its inputs.
    """

    def __init__(
        self,
        *,
        n_rounds: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_leaf: float = 20.0,
        l2: float = 1.0,
        n_bins: int = 32,
    ) -> None:
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.l2 = l2
        self.n_bins = n_bins
        self._edges: list[np.ndarray] = []
        self._trees: list[_Tree] = []
        self._base = 0.0

    def _bin(self, X: np.ndarray) -> np.ndarray:
        codes = np.empty(X.shape, dtype=np.int64)
        for j, edges in enumerate(self._edges):
            codes[:, j] = np.searchsorted(edges, X[:, j], side="right")
        return codes

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> BoostedRegressor:
        """Fit to ``y`` with optional nonnegative row weights.

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix, one row per observation.
        y : np.ndarray
            Regression target.
        sample_weight : np.ndarray | None
            Nonnegative row weights; ``None`` weights every row equally.

        Returns
        -------
        BoostedRegressor
            The fitted model.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones(y.shape) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if X.ndim != 2 or X.shape[0] != y.size or w.shape != y.shape:
            raise ValueError("X, y and sample_weight must align on rows")
        if not (np.isfinite(X).all() and np.isfinite(y).all() and (w >= 0).all()):
            raise ValueError("inputs must be finite with nonnegative weights")
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self._edges = [np.unique(np.quantile(X[:, j], quantiles)) for j in range(X.shape[1])]
        codes = self._bin(X)
        scale = float(w.mean())
        min_leaf = self.min_leaf * scale
        l2 = self.l2 * scale
        self._base = float(np.average(y, weights=w))
        pred = np.full(y.shape, self._base)
        self._trees = []
        for _ in range(self.n_rounds):
            tree = self._grow(codes, y - pred, w, min_leaf, l2)
            self._trees.append(tree)
            pred += self.learning_rate * self._apply(tree, codes)
        return self

    def _grow(
        self, codes: np.ndarray, r: np.ndarray, w: np.ndarray, min_leaf: float, l2: float
    ) -> _Tree:
        feature: list[int] = []
        split_bin: list[int] = []
        left: list[int] = []
        right: list[int] = []
        value: list[float] = []
        n_bins = self.n_bins

        def node(rows: np.ndarray, depth: int) -> int:
            idx = len(feature)
            feature.append(-1)
            split_bin.append(-1)
            left.append(-1)
            right.append(-1)
            g_total = float(w[rows] @ r[rows])
            h_total = float(w[rows].sum())
            value.append(g_total / (h_total + l2))
            if depth >= self.max_depth or h_total < 2.0 * min_leaf:
                return idx
            parent = g_total**2 / (h_total + l2)
            best = (0.0, -1, -1)
            wr = w[rows] * r[rows]
            for j in range(codes.shape[1]):
                c = codes[rows, j]
                g = np.cumsum(np.bincount(c, weights=wr, minlength=n_bins))
                h = np.cumsum(np.bincount(c, weights=w[rows], minlength=n_bins))
                g_right = g_total - g
                h_right = h_total - h
                ok = (h >= min_leaf) & (h_right >= min_leaf)
                if not ok.any():
                    continue
                gain = np.where(ok, g**2 / (h + l2) + g_right**2 / (h_right + l2) - parent, -np.inf)
                b = int(np.argmax(gain))
                if gain[b] > best[0] + 1e-12:
                    best = (float(gain[b]), j, b)
            if best[1] < 0:
                return idx
            _, j, b = best
            goes_left = codes[rows, j] <= b
            feature[idx] = j
            split_bin[idx] = b
            left[idx] = node(rows[goes_left], depth + 1)
            right[idx] = node(rows[~goes_left], depth + 1)
            return idx

        node(np.arange(r.size), 0)
        return _Tree(
            feature=np.array(feature),
            split_bin=np.array(split_bin),
            left=np.array(left),
            right=np.array(right),
            value=np.array(value),
        )

    @staticmethod
    def _apply(tree: _Tree, codes: np.ndarray) -> np.ndarray:
        at = np.zeros(codes.shape[0], dtype=np.int64)
        rows = np.arange(codes.shape[0])
        while True:
            internal = tree.feature[at] >= 0
            if not internal.any():
                return tree.value[at]
            sel = rows[internal]
            nodes = at[sel]
            goes_left = codes[sel, tree.feature[nodes]] <= tree.split_bin[nodes]
            at[sel] = np.where(goes_left, tree.left[nodes], tree.right[nodes])

    def staged_predict(self, X: np.ndarray, rounds: tuple[int, ...]) -> np.ndarray:
        """Predictions after each requested round count, one row per count.

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix to score.
        rounds : tuple[int, ...]
            Increasing round counts no larger than ``n_rounds``.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(rounds), n_rows)``.
        """
        codes = self._bin(np.asarray(X, dtype=float))
        pred = np.full(codes.shape[0], self._base)
        out = np.empty((len(rounds), codes.shape[0]))
        k = 0
        while k < len(rounds) and rounds[k] == 0:
            out[k] = pred
            k += 1
        for t, tree in enumerate(self._trees, start=1):
            pred = pred + self.learning_rate * self._apply(tree, codes)
            while k < len(rounds) and rounds[k] == t:
                out[k] = pred
                k += 1
        if k != len(rounds):
            raise ValueError(f"rounds {rounds} exceed the fitted {len(self._trees)}")
        return out

    def predict(self, X: np.ndarray, n_rounds: int | None = None) -> np.ndarray:
        """Predict after ``n_rounds`` trees (all fitted trees by default).

        Parameters
        ----------
        X : np.ndarray
            Covariate matrix to score.
        n_rounds : int | None
            Number of leading trees to use.

        Returns
        -------
        np.ndarray
            One prediction per row.
        """
        rounds = len(self._trees) if n_rounds is None else n_rounds
        return self.staged_predict(X, (rounds,))[0]


def r_loss(tau: np.ndarray, y: np.ndarray, d: np.ndarray, m: np.ndarray, e: float) -> float:
    """Mean Robinson residual loss of effect predictions ``tau``.

    Parameters
    ----------
    tau : np.ndarray
        Predicted conditional effects.
    y : np.ndarray
        Outcomes.
    d : np.ndarray
        Binary treatment indicators.
    m : np.ndarray
        Outcome-mean predictions not fitted on these rows.
    e : float
        Treated share.

    Returns
    -------
    float
        ``mean(((y - m) - tau (d - e))^2)``.
    """
    return float(np.mean(((y - m) - tau * (d - e)) ** 2))


@dataclass
class RLearner:
    """R-learner whose round count is chosen on a separate selection split.

    ``fit`` uses only fit-split rows: a cross-fitted outcome mean (folds keyed
    on canonical unit or cluster ids) and the treated share. ``select`` picks
    the round count from the predeclared ``ROUNDS_GRID`` by R-loss on the
    selection split; it never sees the final evaluation rows.
    """

    seed: int
    n_folds: int = 5
    rounds_grid: tuple[int, ...] = ROUNDS_GRID
    n_rounds: int | None = None
    e: float = field(init=False, default=0.5)
    _tau_model: BoostedRegressor | None = field(init=False, default=None, repr=False)
    _outcome_model: BoostedRegressor | None = field(init=False, default=None, repr=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        unit_ids: np.ndarray,
        cluster_ids: np.ndarray | None = None,
    ) -> RLearner:
        """Fit the cross-fitted outcome mean and the effect model on fit-split rows.

        Parameters
        ----------
        X : np.ndarray
            Pre-treatment covariate matrix.
        y : np.ndarray
            Outcomes.
        d : np.ndarray
            Binary randomized treatment.
        unit_ids : np.ndarray
            Unit ids; repeated ids stay inside one fold.
        cluster_ids : np.ndarray | None
            Dependence clusters kept inside one fold.

        Returns
        -------
        RLearner
            The fitted learner.
        """
        d = np.asarray(d, dtype=float)
        self.e = float(d.mean())
        folds = fold_assignments(
            unit_ids, n_folds=self.n_folds, seed=self.seed, stratify=d, cluster_ids=cluster_ids
        )
        m = np.empty(y.shape)
        for k in range(self.n_folds):
            train = folds != k
            model = _outcome_regressor().fit(X[train], y[train])
            m[~train] = model.predict(X[~train])
        self._outcome_model = _outcome_regressor().fit(X, y)
        resid = d - self.e
        self._tau_model = _tau_regressor(max(self.rounds_grid)).fit(
            X, (y - m) / resid, sample_weight=resid**2
        )
        return self

    def select(self, X: np.ndarray, y: np.ndarray, d: np.ndarray) -> int:
        """Choose the round count by R-loss on selection-split rows.

        Parameters
        ----------
        X : np.ndarray
            Selection-split covariates.
        y : np.ndarray
            Selection-split outcomes.
        d : np.ndarray
            Selection-split treatment.

        Returns
        -------
        int
            The selected round count, also stored on the learner.
        """
        tau_model, outcome_model = self._fitted()
        m = outcome_model.predict(X)
        staged = tau_model.staged_predict(X, self.rounds_grid)
        losses = [r_loss(t, y, np.asarray(d, dtype=float), m, self.e) for t in staged]
        self.n_rounds = self.rounds_grid[int(np.argmin(losses))]
        return self.n_rounds

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted conditional effects at the selected round count.

        Parameters
        ----------
        X : np.ndarray
            Covariates to score.

        Returns
        -------
        np.ndarray
            One predicted effect per row; no interval is implied.
        """
        tau_model, _ = self._fitted()
        if self.n_rounds is None:
            raise RuntimeError("call select() on the selection split before predict()")
        return tau_model.predict(X, self.n_rounds)

    def _fitted(self) -> tuple[BoostedRegressor, BoostedRegressor]:
        if self._tau_model is None or self._outcome_model is None:
            raise RuntimeError("RLearner used before fit()")
        return self._tau_model, self._outcome_model


def _tau_regressor(n_rounds: int) -> BoostedRegressor:
    # Slow, heavily regularized boosting: pseudo-outcomes are noisy.
    return BoostedRegressor(
        n_rounds=n_rounds, learning_rate=0.02, max_depth=3, min_leaf=200.0, l2=10.0
    )


def _outcome_regressor() -> BoostedRegressor:
    return BoostedRegressor(n_rounds=100, learning_rate=0.1, max_depth=3)
