"""Unit tests for honest-split CATE validation and the targeting rule on top of it."""

from __future__ import annotations

import functools
import itertools
import math
from typing import Any, cast

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import norm

from increment.errors import IncrementWarning, InvalidRequestError
from increment.estimation._adjust.encoding import CovariateLayout
from increment.estimation._adjust.learners import LogisticPropensity, RidgeOutcome
from increment.estimation._score_design import PsiFn, ScoreDesign
from increment.estimation.cate import Covariate, fit_cate
from increment.estimation.targeting import (
    TargetingSelection,
    _clan,
    _curve_weights,
    _dr_psi,
    _holdout_mask,
    _honest_holdout,
    _ipw_psi,
    _rank_test,
    _sorted_groups,
    _unit_weights,
    _validation,
    select_targeting_rule_arrays,
    targeting_rule_arrays,
    validate_cate_arrays,
)
from increment.semantics.design import IdentificationGate

Z95 = 1.959963984540054  # norm.ppf(0.975)

# crc32("u0") % 2 == 0 .. an id sequence that happens to split 20/20 and 40/40.
IDS_40 = np.array([f"u{i}" for i in range(40)])
IDS_80 = np.array([f"u{i}" for i in range(80)])

PLATFORM = Covariate(name="platform", kind="categorical")
SPEND = Covariate(name="spend")


def _content_hash_holdout(
    unit_ids: np.ndarray, cluster_ids: np.ndarray | None = None
) -> np.ndarray:
    """The documented split, recomputed from the definition: ``crc32(id) % 2 == 1``.

    Keyed on the cluster id when supplied, so every member of a cluster lands
    on the same side.  Callers that need to know which rows the pipeline held
    out build the expectation from this rather than from the module's own
    partition helper.
    """
    import zlib

    keys = unit_ids if cluster_ids is None else cluster_ids
    return np.array([zlib.crc32(str(u).encode()) % 2 == 1 for u in keys], dtype=bool)


def _tied_fixture() -> dict:
    """40 units, a two-level categorical the only effect modifier.

    Every unit on a platform gets the SAME predicted effect, so the holdout
    score is one big pair of ties - the case where rank weighting has to be
    tie-aware or the answer moves with the row order.
    """
    platform = np.array(["ios", "android"] * 20)
    # Treatment cycles at a different period than platform so the two are
    # not collinear in the training half.
    d = np.array([(i // 2) % 2 for i in range(40)], dtype=float)
    wobble = np.array([((i * 37) % 11 - 5) / 5 for i in range(40)])
    ios = (platform == "ios").astype(float)
    y = 1.0 + 0.4 * ios + d * (0.5 + 1.2 * ios) + wobble
    return {"y": y, "d": d, "cols": {"platform": platform}, "unit_ids": IDS_40}


def _signal_fixture() -> dict:
    """80 units with a real linear CATE in ``spend`` plus seeded noise."""
    spend = np.linspace(-2.0, 2.0, 80)
    platform = np.array(["ios", "android"] * 40)
    d = np.array([i % 2 for i in range(80)], dtype=float)
    noise = np.random.default_rng(6).standard_normal(80)
    y = 1.0 + 0.5 * spend + 0.3 * (platform == "ios") + d * (0.5 + 0.8 * spend) + 1.5 * noise
    return {
        "y": y,
        "d": d,
        "cols": {"spend": spend, "platform": platform},
        "unit_ids": IDS_80,
    }


def _call_array_entry_point(caller: str, fixture: dict, **kwargs):
    if caller == "validate_cate_arrays":
        return validate_cate_arrays(**fixture, **kwargs)
    if caller == "targeting_rule_arrays":
        return targeting_rule_arrays(**fixture, fraction=0.4, **kwargs)
    return select_targeting_rule_arrays(
        **fixture,
        fractions=(0.2, 0.4),
        n_folds=2,
        seed=11,
        **kwargs,
    )


@pytest.mark.parametrize(
    "caller", ["validate_cate_arrays", "targeting_rule_arrays", "select_targeting_rule_arrays"]
)
def test_array_entry_points_refuse_unadjustable_dtype_before_fit(caller, monkeypatch):
    fixture = _tied_fixture()
    fixture["cols"]["joined"] = np.array(["2024-01-01"] * 40, dtype="datetime64[D]")

    def fail(*_args, **_kwargs):
        raise AssertionError("CATE model fitted before validating observational adjustment")

    monkeypatch.setattr("increment.estimation.targeting.fit_cate", fail)
    with pytest.raises(InvalidRequestError) as exc_info:
        _call_array_entry_point(caller, fixture, interact=[PLATFORM], adjustment=("joined",))
    assert exc_info.value.code == "estimation.targeting.adjustment_dtype"
    assert exc_info.value.context["caller"] == caller
    assert exc_info.value.context["column"] == "joined"


def _training_side(caller: str, unit_ids: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The rows each entry point fits on, from the documented split definitions:
    the content-hash complement, or the seeded arm-stratified inner half that
    `_call_array_entry_point` requests of the nested selection."""
    if caller == "select_targeting_rule_arrays":
        from increment.estimation.crossfit import outer_split

        return ~outer_split(unit_ids, test_size=0.5, seed=11, stratify=d, cluster_ids=None)
    return ~_content_hash_holdout(unit_ids)


def _validation_of(result: Any) -> Any:
    """The honest-split validation each entry point reports."""
    rule = result.rule if isinstance(result, TargetingSelection) else result
    return getattr(rule, "validation", rule)


def _opposite_modes_fixture(caller: str) -> tuple[dict, np.ndarray]:
    """`_signal_fixture` whose ``platform`` is modal ``ios`` on the training
    side and modal ``android`` on the scored side, each level present on
    both: a basis fitted on the rows it scores would flip its reference
    level, and with it the meaning of the one indicator column, between
    the partitions one workflow scores."""
    fixture = _signal_fixture()
    train = _training_side(caller, fixture["unit_ids"], fixture["d"])
    platform = np.where(train, "ios", "android").astype(object)
    platform[np.flatnonzero(train)[::4]] = "android"
    platform[np.flatnonzero(~train)[::4]] = "ios"
    fixture["cols"]["platform"] = platform.astype(str)
    return fixture, train


def _android_scorer(column: str, seen: list[tuple[ScoreDesign, np.ndarray]]) -> PsiFn:
    """A score reading the android indicator by name, recording what it saw."""

    def score(
        y: np.ndarray,
        d: np.ndarray,
        X: ScoreDesign,
        unit_ids: np.ndarray,
        cluster_ids: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del d, cluster_ids
        seen.append((X, unit_ids))
        android = X[:, X.columns.index(column)]
        return np.asarray(y, dtype=float) + 3.0 * android, np.ones(y.shape, dtype=bool)

    return score


@pytest.mark.parametrize(
    "caller", ["validate_cate_arrays", "targeting_rule_arrays", "select_targeting_rule_arrays"]
)
def test_custom_scores_read_one_basis_fitted_on_training_rows_in_every_call(caller):
    """A caller-supplied score reads each categorical through one basis
    fitted on the training side alone -- its modal level the reference --
    in every call of a workflow, so the same level always means the same
    named column, whichever level is modal among the rows scored. Reading
    that column by name reproduces the hand-built dummy oracle exactly."""
    from tests.categorical_cases import assert_rows_match

    fixture, train = _opposite_modes_fixture(caller)
    platform = fixture["cols"]["platform"]
    android = (platform == "android").astype(float)
    assert (platform[train] == "ios").sum() > (platform[train] == "android").sum()
    assert (platform[~train] == "android").sum() > (platform[~train] == "ios").sum()

    oracle_fixture = {**fixture, "cols": {**fixture["cols"], "platform_android": android}}
    oracle = _call_array_entry_point(
        caller,
        oracle_fixture,
        interact=[SPEND],
        adjustment=("spend", "platform_android"),
        psi_fn=_android_scorer("platform_android", []),
        arm_summary="score",
    )
    seen: list[tuple[ScoreDesign, np.ndarray]] = []
    actual = _call_array_entry_point(
        caller,
        fixture,
        interact=[SPEND],
        adjustment=("spend", "platform"),
        psi_fn=_android_scorer("platform=android", seen),
        arm_summary="score",
    )

    expected_ids = fixture["unit_ids"]
    if caller != "select_targeting_rule_arrays":
        expected_ids = expected_ids[~train]
    assert {unit for _, unit_ids in seen for unit in unit_ids} == set(expected_ids)
    positions = {unit: i for i, unit in enumerate(fixture["unit_ids"])}
    for matrix, unit_ids in seen:
        assert matrix.dtype == np.float64 and matrix.shape[1] == 2
        assert matrix.columns == ("spend", "platform=android")
        assert matrix.sources == ("spend", "platform")
        rows = np.array([positions[str(unit)] for unit in unit_ids])
        np.testing.assert_array_equal(matrix[:, 0], fixture["cols"]["spend"][rows])
        np.testing.assert_array_equal(matrix[:, 1], android[rows])
    assert_rows_match(oracle.model_dump(), actual.model_dump(), skip=("required_columns",))


@pytest.mark.parametrize(
    "caller", ["validate_cate_arrays", "targeting_rule_arrays", "select_targeting_rule_arrays"]
)
def test_custom_scores_read_a_scored_level_absent_from_training_as_the_reference(caller):
    """A level only the scored side carries never trains the basis: the
    score still sees the training-fitted columns, that row reads as the
    reference (every indicator zero), the coded advisory names the row and
    the caller-supplied score, and nothing is trimmed -- exactly what a
    hand-built dummy column identically zero on that row would give."""
    import warnings

    from tests.categorical_cases import assert_rows_match

    fixture, train = _opposite_modes_fixture(caller)
    platform = fixture["cols"]["platform"].astype(object)
    lone = int(np.flatnonzero(~train)[0])
    platform[lone] = "web"
    fixture["cols"]["platform"] = platform.astype(str)
    android = (fixture["cols"]["platform"] == "android").astype(float)

    oracle_fixture = {**fixture, "cols": {**fixture["cols"], "platform_android": android}}
    oracle = _call_array_entry_point(
        caller,
        oracle_fixture,
        interact=[SPEND],
        adjustment=("spend", "platform_android"),
        psi_fn=_android_scorer("platform_android", []),
        arm_summary="score",
    )
    seen: list[tuple[ScoreDesign, np.ndarray]] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = _call_array_entry_point(
            caller,
            fixture,
            interact=[SPEND],
            adjustment=("spend", "platform"),
            psi_fn=_android_scorer("platform=android", seen),
            arm_summary="score",
        )

    (advisory,) = [
        w.message
        for w in caught
        if getattr(w.message, "code", None) == "estimation.targeting.unseen_level_advisory"
    ]
    assert isinstance(advisory, IncrementWarning)
    assert advisory.context["unseen"] == (("platform", "web", 1, 1),)
    assert advisory.context["n"] == int((~train).sum())
    assert advisory.context["score"] == "caller-supplied score"
    positions = {unit: i for i, unit in enumerate(fixture["unit_ids"])}
    for matrix, unit_ids in seen:
        assert matrix.columns == ("spend", "platform=android")
        rows = np.array([positions[str(unit)] for unit in unit_ids])
        np.testing.assert_array_equal(matrix[:, 1], android[rows])
    scored_rows = {positions[str(unit)] for unit in seen[-1][1]}
    assert lone in scored_rows and len(scored_rows) == int((~train).sum())
    validation = _validation_of(actual)
    assert validation.population is None
    assert validation.n_holdout == int((~train).sum())
    assert_rows_match(oracle.model_dump(), actual.model_dump(), skip=("required_columns",))


def test_score_design_names_follow_rows_and_leave_arithmetic_plain():
    """The design a custom score reads is an ndarray whose column names ride
    along with row selection and copies, are empty on a narrower view, and
    never leak into arithmetic or reductions."""
    from increment.estimation._score_design import ScoreDesign

    matrix = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    design = ScoreDesign(matrix, ("spend", "platform=android"), ("spend", "platform"))
    assert isinstance(design, np.ndarray) and design.dtype == np.float64
    assert design.columns == ("spend", "platform=android")
    assert design.sources == ("spend", "platform")

    rows = design[np.array([True, False, True])]
    assert rows.columns == design.columns and rows.sources == design.sources
    assert design[[0, 2], ...].columns == design.columns
    assert design.copy().columns == design.columns
    assert design[:, :1].columns == () and design[:, :1].sources == ()
    assert design[:, ::-1].columns == () and design[:, ::-1].sources == ()
    assert design[:2].T.columns == () and design[:2].T.sources == ()

    android = design[:, design.columns.index("platform=android")]
    np.testing.assert_array_equal(android, [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(android * 2.0, [0.0, 2.0, 0.0])
    np.testing.assert_array_equal(np.asarray(design), matrix)
    assert design.mean() == pytest.approx(matrix.mean())
    np.testing.assert_array_equal(design.sum(axis=1), [1.0, 3.0, 3.0])


@pytest.mark.parametrize(
    "caller", ["validate_cate_arrays", "targeting_rule_arrays", "select_targeting_rule_arrays"]
)
def test_array_entry_points_accept_numeric_adjustment_only_with_categorical_cate(caller):
    n = 80
    platform = np.array(["ios", "android"] * (n // 2))
    d = np.array([(i // 2) % 2 for i in range(n)], dtype=float)
    ios = (platform == "ios").astype(float)
    fixture = {
        "y": 1.0 + 0.4 * ios + d * (0.5 + 1.2 * ios),
        "d": d,
        "cols": {"platform": platform, "spend": np.linspace(-2.0, 2.0, n)},
        "unit_ids": IDS_80,
    }
    seen: list[np.ndarray] = []

    def score(y, d, X, unit_ids, cluster_ids):
        del d, unit_ids, cluster_ids
        seen.append(X.copy())
        return np.asarray(y, dtype=float), np.ones(y.shape, dtype=bool)

    _call_array_entry_point(
        caller,
        fixture,
        interact=[PLATFORM],
        adjustment=("spend",),
        psi_fn=score,
        arm_summary="score",
    )
    assert seen
    assert all(matrix.dtype.kind == "f" and matrix.shape[1] == 1 for matrix in seen)


@pytest.mark.parametrize(
    "caller", ["validate_cate_arrays", "targeting_rule_arrays", "select_targeting_rule_arrays"]
)
@pytest.mark.parametrize("object_kind", ["decimal", "pandas_nullable"])
def test_array_entry_points_normalize_numeric_object_adjustments(caller, object_kind):
    from decimal import Decimal

    fixture = _signal_fixture()
    if object_kind == "decimal":
        fixture["cols"]["spend"] = np.array(
            [Decimal(str(value)) for value in fixture["cols"]["spend"]], dtype=object
        )
    else:
        import pandas as pd

        fixture["cols"]["spend"] = pd.array(np.arange(IDS_80.size), dtype="Int64").to_numpy(
            dtype=object
        )
    seen: list[np.ndarray] = []

    def score(y, d, X, unit_ids, cluster_ids):
        del d, unit_ids, cluster_ids
        seen.append(X.copy())
        return np.asarray(y, dtype=float), np.ones(y.shape, dtype=bool)

    _call_array_entry_point(
        caller,
        fixture,
        interact=[SPEND],
        adjustment=("spend",),
        psi_fn=score,
        arm_summary="score",
    )
    assert seen
    assert all(matrix.dtype == np.float64 and matrix.shape[1] == 1 for matrix in seen)


@pytest.mark.parametrize(
    "caller", ["validate_cate_arrays", "targeting_rule_arrays", "select_targeting_rule_arrays"]
)
def test_array_entry_points_treat_numeric_strings_as_levels_not_numbers(caller):
    """A string column is categorical whatever its strings spell: ``"-2.0"``
    is a level, never parsed into a number, so a custom score sees one
    indicator per non-modal distinct string of the training rows, named by
    the string, instead of one numeric column."""
    fixture = _signal_fixture()
    spelled = np.array(["0.000", "-2.0", "1e3", "1e3"], dtype=object)
    fixture["cols"]["tier"] = spelled[np.arange(80) % 4]
    seen: list[ScoreDesign] = []

    def score(y, d, X, unit_ids, cluster_ids):
        del d, unit_ids, cluster_ids
        seen.append(X.copy())
        return np.asarray(y, dtype=float), np.ones(y.shape, dtype=bool)

    _call_array_entry_point(
        caller, fixture, interact=[SPEND], adjustment=("tier",), psi_fn=score, arm_summary="score"
    )
    assert seen
    for matrix in seen:
        assert matrix.dtype == np.float64
        assert set(np.unique(matrix).tolist()) <= {0.0, 1.0}
        # Three strings are three levels: the modal training one is the
        # reference, each other one an indicator named by its string.
        assert len(matrix.columns) == 2
        assert set(matrix.columns) < {"tier=0.000", "tier=-2.0", "tier=1e3"}
        assert matrix.sources == ("tier", "tier")
        assert matrix.sum(axis=1).max() == 1.0


@pytest.mark.parametrize("object_kind", ["decimal", "pandas_nullable"])
def test_array_numeric_object_adjustment_preserves_null_refusal(object_kind):
    from decimal import Decimal

    fixture = _signal_fixture()
    if object_kind == "decimal":
        values = np.array([Decimal(str(value)) for value in fixture["cols"]["spend"]], dtype=object)
        values[7] = None
    else:
        import pandas as pd

        values = pd.array(np.arange(IDS_80.size), dtype="Int64")
        values[7] = pd.NA
        values = values.to_numpy(dtype=object)
    fixture["cols"]["spend"] = values

    with pytest.raises(InvalidRequestError) as exc_info:
        validate_cate_arrays(
            **fixture,
            interact=[SPEND],
            adjustment=("spend",),
        )
    assert exc_info.value.code == "estimation.cate.covariate_nulls"
    assert exc_info.value.context["name"] == "spend"


def _step_fixture() -> dict:
    """80 NOISELESS units: the effect is 0.5 below zero spend and 2.5 above.

    The fitted score is monotone in spend and the holdout happens to split
    20/20 around zero, so the top 40% of it is exactly the high-effect
    step: every treated unit in there reads 3.5 and every control 1.0, and
    the policy value is 2.5 with no arithmetic left to do.  AUTOC lands at
    p = 0.0254, so the one fixture passes the gate at 5% and fails it at
    1% - both branches of the recommendation off the same data.
    """
    spend = np.linspace(-2.0, 2.0, 80)
    d = np.array([i % 2 for i in range(80)], dtype=float)
    y = 1.0 + d * (0.5 + 2.0 * (spend > 0.0))
    return {"y": y, "d": d, "cols": {"spend": spend}, "unit_ids": IDS_80}


def _refit_holdout(fixture: dict, interact) -> tuple[np.ndarray, np.ndarray]:
    """The holdout score and psi, rebuilt from the fixture from scratch.

    Independent of whatever the module did internally, so the numbers it
    reports are checked against the definition rather than against
    themselves.
    """
    hold = _content_hash_holdout(fixture["unit_ids"])
    train = ~hold
    fit = fit_cate(
        fixture["y"][train],
        fixture["d"][train],
        {name: values[train] for name, values in fixture["cols"].items()},
        interact=interact,
    )
    score = fit.score(
        {name: values[hold] for name, values in fixture["cols"].items()}, deploy_grain="unit"
    )
    return score, _ipw_psi(fixture["y"][hold], fixture["d"][hold])


def _flatten(v) -> list[float]:
    """Every number a validation reports, for order-invariance comparison."""
    assert v.holdout_ate is not None
    out = [
        float(v.n_train),
        float(v.n_holdout),
        v.holdout_ate.value,
        v.autoc.estimate,
        v.autoc.se,
        v.autoc.p_value,
        v.qini.estimate,
        v.qini.se,
        v.qini.p_value,
    ]
    for g in v.groups:
        out += [float(g.group), float(g.n), g.mean_score, g.effect, g.se, g.lb, g.ub]
    for r in v.clan:
        out += [r.mean_most, r.mean_least, r.diff, r.se, r.lb, r.ub]
    return out


class TestGuardedDesignRemainingRefusals:
    """Representative regression coverage for the design-refusal codes this
    file's other tests don't already exercise, driven through the public
    array entry point that reaches them."""

    @staticmethod
    def _select(y, d, cols, unit_ids, **kwargs):
        return select_targeting_rule_arrays(
            y,
            d,
            cols,
            unit_ids,
            interact=[SPEND],
            fractions=(0.2,),
            n_folds=2,
            seed=11,
            **kwargs,
        )

    def test_non_1d_outcome_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            self._select(np.zeros((4, 1)), np.zeros(4), {"spend": np.zeros(4)}, IDS_40[:4])
        assert exc_info.value.code == "estimation.targeting.outcome_shape"

    def test_mismatched_treatment_shape_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            self._select(np.zeros(4), np.zeros(5), {"spend": np.zeros(4)}, IDS_40[:4])
        assert exc_info.value.code == "estimation.targeting.treatment_shape_expected"

    def test_mismatched_unit_ids_shape_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            self._select(np.zeros(4), np.zeros(4), {"spend": np.zeros(4)}, IDS_40[:5])
        assert exc_info.value.code == "estimation.targeting.unit_ids_shape"

    def test_non_finite_outcome_is_refused(self):
        y = np.array([0.0, np.nan, 1.0, 0.0])
        d = np.array([1.0, 0.0, 1.0, 0.0])
        with pytest.raises(InvalidRequestError) as exc_info:
            self._select(y, d, {"spend": np.zeros(4)}, IDS_40[:4])
        assert exc_info.value.code == "estimation.cate.outcome_nulls_non"

    def test_alpha_outside_unit_interval_is_refused(self):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        d = np.array([1.0, 0.0, 1.0, 0.0])
        with pytest.raises(InvalidRequestError) as exc_info:
            self._select(y, d, {"spend": np.zeros(4)}, IDS_40[:4], alpha=1.5)
        assert exc_info.value.code == "estimation.diagnostics.alpha"

    def test_n_folds_below_two_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            select_targeting_rule_arrays(
                **_selection_fixture(), interact=[SPEND], fractions=(0.2,), n_folds=1, seed=11
            )
        assert exc_info.value.code == "estimation.crossfit.n_folds_least"


class TestSplit:
    def test_parity_one_is_held_out(self):
        import zlib

        expected = [zlib.crc32(u.encode()) % 2 == 1 for u in IDS_40]
        assert _holdout_mask(IDS_40).tolist() == expected

    def test_split_ignores_row_order(self):
        order = np.random.default_rng(0).permutation(40)
        assert _holdout_mask(IDS_40[order]).tolist() == _holdout_mask(IDS_40)[order].tolist()

    def test_integer_unit_ids_are_refused(self):
        """`str(np.int64(1))` and `str(np.float64(1.0))` hash differently."""
        fixture = _signal_fixture()
        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate_arrays(
                fixture["y"], fixture["d"], fixture["cols"], np.arange(80), interact=[SPEND]
            )
        assert exc_info.value.code == "estimation.targeting.unit_ids_string"

    def test_injected_holdout_mask_shape_mismatch_is_refused(self):
        fixture = _signal_fixture()
        with pytest.raises(InvalidRequestError) as exc_info:
            _honest_holdout(
                fixture["y"],
                fixture["d"],
                fixture["cols"],
                IDS_80,
                interact=[SPEND],
                adjust=(),
                n_groups=4,
                alpha=0.05,
                holdout_mask=np.zeros(5, dtype=bool),
            )
        assert exc_info.value.code == "estimation.targeting.holdout_mask_shape"

    def test_validation_refuses_float_unit_ids(self):
        fixture = _signal_fixture()
        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate_arrays(
                fixture["y"],
                fixture["d"],
                fixture["cols"],
                np.arange(80.0),
                interact=[SPEND],
            )
        assert exc_info.value.code == "estimation.targeting.unit_ids_string"

    def test_duplicate_unit_ids_are_refused(self):
        """A duplicated unit_id can land on both sides of the honest
        split (or twice on one side), breaking the split's independence
        assumption; validation should catch this before hashing."""
        fixture = _signal_fixture()
        duped_ids = IDS_80.copy()
        duped_ids[1] = duped_ids[0]  # "u1" -> "u0": one real duplicate
        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate_arrays(
                fixture["y"], fixture["d"], fixture["cols"], duped_ids, interact=[SPEND]
            )
        assert exc_info.value.code == "estimation.targeting.unit_ids_contains"

    def test_every_reported_number_is_row_order_invariant_under_ties(self):
        """A categorical-only score ties everything; midweights carry this."""
        fixture = _tied_fixture()
        base = validate_cate_arrays(**fixture, interact=[PLATFORM], n_groups=2)
        order = np.random.default_rng(3).permutation(40)
        shuffled = validate_cate_arrays(
            fixture["y"][order],
            fixture["d"][order],
            {"platform": fixture["cols"]["platform"][order]},
            IDS_40[order],
            interact=[PLATFORM],
            n_groups=2,
        )
        assert _flatten(shuffled) == pytest.approx(_flatten(base), abs=1e-12, nan_ok=True)

    def test_tied_scores_share_the_mean_rank_weight(self):
        """Ranks 1-2 and 3-4 each collapse to their average AUTOC weight."""
        weights = _unit_weights(np.array([1.0, 1.0, 2.0, 2.0]), np.ones(4))
        top = (13 / 48 + 1 / 48) / 2
        assert weights.tolist() == pytest.approx([-top, -top, top, top])

    def test_each_half_needs_both_arms(self):
        fixture = _signal_fixture()
        hold = _content_hash_holdout(IDS_80)
        d = fixture["d"].copy()
        d[hold] = 0.0
        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate_arrays(fixture["y"], d, fixture["cols"], IDS_80, interact=[SPEND])
        assert exc_info.value.code == "estimation.targeting.half_split_holds"


@pytest.mark.parametrize(
    "caller", ["validate_cate_arrays", "targeting_rule_arrays", "select_targeting_rule_arrays"]
)
def test_targeting_identity_collisions_are_coded_before_fit(caller, monkeypatch):
    fixture = _signal_fixture()
    fixture["unit_ids"] = fixture["unit_ids"].astype(object)
    fixture["unit_ids"][:2] = [1, "1"]

    def fail(*_args, **_kwargs):
        raise AssertionError("fitted before rejecting ambiguous identities")

    monkeypatch.setattr("increment.estimation.targeting.fit_cate", fail)
    with pytest.raises(InvalidRequestError) as exc_info:
        _call_array_entry_point(caller, fixture, interact=[SPEND])
    assert exc_info.value.code == "estimation.crossfit.identity_collision"
    assert exc_info.value.context["what"] == "unit_ids"


def test_selection_rejects_numeric_cluster_ids_before_fit(monkeypatch):
    fixture = _signal_fixture()

    def fail(*_args, **_kwargs):
        raise AssertionError("fitted before rejecting unstable cluster dtype")

    monkeypatch.setattr("increment.estimation.targeting.fit_cate", fail)
    with pytest.raises(InvalidRequestError) as exc_info:
        _call_array_entry_point(
            "select_targeting_rule_arrays",
            fixture,
            interact=[SPEND],
            cluster_ids=np.repeat(np.arange(20), 4),
        )
    assert exc_info.value.code == "estimation.targeting.cluster_ids_string"


class TestClusterSupport:
    """``cluster_ids`` threads through the honest split, the DR fold path,
    and caller-supplied masks without ever splitting a cluster - and
    without changing the pipeline's answer when rows are permuted."""

    def test_holdout_mask_broadcasts_to_every_cluster_member(self):
        cluster_ids = np.repeat([f"c{i}" for i in range(20)], 4)
        hold = _holdout_mask(IDS_80, cluster_ids=cluster_ids)
        for c in np.unique(cluster_ids):
            assert len(set(hold[cluster_ids == c].tolist())) == 1

    def test_psi_callback_alignment_and_retained_population(self):
        fixture = _signal_fixture()
        cluster_ids = np.repeat([f"c{i}" for i in range(20)], 4)
        hold = _content_hash_holdout(IDS_80, cluster_ids=cluster_ids)
        spend = fixture["cols"]["spend"]
        retained = hold & (spend > -0.5)
        expected_psi = (fixture["y"] + fixture["d"] + spend)[retained]
        assert 0 < retained.sum() < hold.sum()

        def score(y, d, X, unit_ids, clusters):
            np.testing.assert_array_equal(y, fixture["y"][hold])
            np.testing.assert_array_equal(d, fixture["d"][hold])
            np.testing.assert_array_equal(X, spend[hold, None])
            np.testing.assert_array_equal(unit_ids, IDS_80[hold])
            np.testing.assert_array_equal(clusters, cluster_ids[hold])
            kept = X[:, 0] > -0.5
            # Excluded scores must not contribute to any reported estimate.
            psi = np.where(kept, y + d + X[:, 0], -1_000.0)
            return psi, kept

        result = validate_cate_arrays(
            **fixture,
            cluster_ids=cluster_ids,
            interact=[SPEND],
            adjustment=("spend",),
            psi_fn=score,
            arm_summary="score",
            n_groups=2,
        )

        expected_effect = expected_psi.mean()
        assert result.n_train == (~hold).sum()
        assert result.n_holdout == retained.sum()
        assert sum(group.n for group in result.groups) == retained.sum()
        assert result.population == "overlap_subpopulation"
        assert result.holdout_ate is not None
        assert result.holdout_ate.value == pytest.approx(expected_effect)

    def test_honest_holdout_default_split_stays_cluster_atomic(self):
        fixture = _signal_fixture()
        cluster_ids = np.repeat([f"c{i}" for i in range(20)], 4)
        holdout = _honest_holdout(
            fixture["y"],
            fixture["d"],
            fixture["cols"],
            IDS_80,
            cluster_ids=cluster_ids,
            interact=[SPEND],
            adjust=(),
            n_groups=4,
            alpha=0.05,
        )
        assert holdout.cluster_ids is not None
        assert holdout.cluster_ids.size == holdout.y.size
        # Every cluster is wholly in or wholly out of the holdout - checked
        # against the full-length inputs, not the already-filtered holdout
        # array, which would trivially agree with itself either way.
        hold = _content_hash_holdout(IDS_80, cluster_ids=cluster_ids)
        for c in np.unique(cluster_ids):
            assert len(set(hold[cluster_ids == c].tolist())) == 1
        # A cluster that lands in the holdout carries every one of its
        # original members, not a partial subset.
        full_counts = {c: int((cluster_ids == c).sum()) for c in np.unique(cluster_ids)}
        for c in np.unique(holdout.cluster_ids):
            assert int((holdout.cluster_ids == c).sum()) == full_counts[c]

    def test_honest_holdout_refuses_a_supplied_mask_that_splits_a_cluster(self):
        fixture = _signal_fixture()
        cluster_ids = np.repeat([f"c{i}" for i in range(20)], 4)
        bad_mask = np.zeros(80, dtype=bool)
        bad_mask[0] = True  # cluster c0's first member is held out
        bad_mask[1] = False  # c0's second member stays in training -- a split
        bad_mask[40:] = True  # fill the rest of the holdout half
        with pytest.raises(InvalidRequestError) as exc_info:
            _honest_holdout(
                fixture["y"],
                fixture["d"],
                fixture["cols"],
                IDS_80,
                cluster_ids=cluster_ids,
                interact=[SPEND],
                adjust=(),
                n_groups=4,
                alpha=0.05,
                holdout_mask=bad_mask,
            )
        assert exc_info.value.code == "estimation.crossfit.cluster_split"

    def test_randomized_training_mask_cannot_remove_one_member_of_a_cluster(self):
        fixture = _signal_fixture()
        ids = np.repeat([f"c{i}" for i in range(20)], 4)
        hold = _content_hash_holdout(IDS_80, cluster_ids=ids)
        training = ~hold
        training[np.flatnonzero(training)[0]] = False
        with pytest.raises(InvalidRequestError) as caught:
            _honest_holdout(
                **fixture,
                cluster_ids=ids,
                interact=[SPEND],
                adjust=(),
                n_groups=2,
                alpha=0.05,
                training_mask=training,
            )
        assert caught.value.code == "estimation.crossfit.cluster_split"

    def test_cluster_ids_shape_mismatch_is_refused(self):
        fixture = _signal_fixture()
        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate_arrays(
                fixture["y"],
                fixture["d"],
                fixture["cols"],
                IDS_80,
                cluster_ids=np.arange(5),
                interact=[SPEND],
            )
        assert exc_info.value.code == "estimation.targeting.cluster_ids_shape"

    @staticmethod
    def _mixed_treatment_clustered_fixture(
        seed: int = 2, n_clusters: int = 40, per_cluster: int = 10
    ):
        rng = np.random.default_rng(seed)
        cluster_ids = np.repeat([f"clu{i}" for i in range(n_clusters)], per_cluster)
        n = cluster_ids.size
        spend = rng.normal(size=n)
        d = rng.binomial(1, expit(0.5 * spend)).astype(float)
        y = 1.0 + spend + d * (1.0 + 0.5 * spend) + rng.normal(0.0, 0.5, n)
        unit_ids = np.array([f"u{i}" for i in range(n)])
        return y, d, spend, unit_ids, cluster_ids

    @staticmethod
    def _dr_psi_fn():
        return functools.partial(
            _dr_psi,
            propensity_learner=LogisticPropensity,
            outcome_learner=RidgeOutcome,
            folds=4,
            seed=3,
            gate=IdentificationGate(overlap="trim"),
        )

    def test_mixed_treatment_clustered_dr_selection_is_row_order_invariant(self):
        """The whole pipeline - outer split, cross-fitted DR folds, and the
        inner selection folds - keys on canonical cluster identity, not row
        position: permuting every row while keeping (y, d, spend, unit_ids,
        cluster_ids) aligned must not move the answer. A leaked cluster or
        a positional fold assignment anywhere in the chain would make this
        numerically unstable across the permutation."""
        y, d, spend, unit_ids, cluster_ids = self._mixed_treatment_clustered_fixture()
        select = functools.partial(
            select_targeting_rule_arrays,
            interact=[SPEND],
            fractions=(0.2, 0.4),
            n_folds=4,
            seed=5,
            adjustment=("spend",),
            arm_summary="score",
            psi_fn=self._dr_psi_fn(),
        )
        base = select(y, d, {"spend": spend}, unit_ids, cluster_ids=cluster_ids)
        order = np.random.default_rng(9).permutation(unit_ids.size)
        shuffled = select(
            y[order],
            d[order],
            {"spend": spend[order]},
            unit_ids[order],
            cluster_ids=cluster_ids[order],
        )
        assert shuffled.selected_fraction == base.selected_fraction
        assert base.rule.policy_value is not None
        assert shuffled.rule.policy_value is not None
        assert shuffled.rule.policy_value.value == pytest.approx(base.rule.policy_value.value)


class TestSortedGroups:
    """20 units, score ``i``, effect exactly ``0.2 * i``.

    Quartiles cut at 4.75 / 9.5 / 14.25, so each group holds five
    consecutive units with mean index 2, 7, 12 and 17; the group effects
    must therefore be 0.4, 1.4, 2.4 and 3.4 exactly.
    """

    score = np.arange(20.0)
    d = np.array([i % 2 for i in range(20)], dtype=float)
    y = 0.5 * np.arange(20.0) + 0.2 * np.arange(20.0) * d

    def test_group_effects_match_the_hand_computation(self):
        groups = _sorted_groups(
            self.score, self.y, self.d, _ipw_psi(self.y, self.d), 4, Z95, arm_summary="welch"
        )
        assert [g.group for g in groups] == [1, 2, 3, 4]
        assert [g.n for g in groups] == [5, 5, 5, 5]
        assert [g.mean_score for g in groups] == [2.0, 7.0, 12.0, 17.0]
        assert [g.effect for g in groups] == pytest.approx([0.4, 1.4, 2.4, 3.4])

    def test_group_standard_errors_match_the_welch_formula(self):
        groups = _sorted_groups(
            self.score, self.y, self.d, _ipw_psi(self.y, self.d), 4, Z95, arm_summary="welch"
        )
        # Group 1: treated y = 0.7, 2.1 (var 0.98); control y = 0, 1, 2 (var 1).
        assert groups[0].se == pytest.approx(math.sqrt(0.98 / 2 + 1.0 / 3))
        assert groups[0].effect is not None and groups[0].se is not None
        assert groups[0].lb == pytest.approx(groups[0].effect - Z95 * groups[0].se)
        assert groups[0].ub == pytest.approx(groups[0].effect + Z95 * groups[0].se)

    def test_a_group_thin_in_one_arm_is_counted_with_nan_effects(self):
        d = np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=float)
        y = np.arange(8.0)
        groups = _sorted_groups(np.arange(8.0), y, d, _ipw_psi(y, d), 2, Z95, arm_summary="welch")
        assert groups[0].n == 4  # counted, not dropped
        assert groups[0].effect is not None and groups[0].se is not None
        assert groups[0].lb is not None and groups[0].ub is not None
        assert math.isnan(groups[0].effect) and math.isnan(groups[0].se)
        assert math.isnan(groups[0].lb) and math.isnan(groups[0].ub)
        assert groups[1].effect is not None
        assert not math.isnan(groups[1].effect)

    def test_a_fully_tied_score_leaves_the_upper_group_empty(self):
        d = np.array([0, 1, 0, 1, 0, 1], dtype=float)
        y = np.arange(6.0)
        groups = _sorted_groups(np.ones(6), y, d, _ipw_psi(y, d), 2, Z95, arm_summary="welch")
        assert [g.n for g in groups] == [6, 0]
        assert groups[1].mean_score is not None and groups[1].effect is not None
        assert math.isnan(groups[1].mean_score) and math.isnan(groups[1].effect)

    def test_n_groups_below_two_is_refused(self):
        fixture = _signal_fixture()
        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate_arrays(**fixture, interact=[SPEND], n_groups=1)
        assert exc_info.value.code == "estimation.targeting.n_groups_least"


def _brute_force_area(score: np.ndarray, psi: np.ndarray, u: np.ndarray) -> float:
    """``(1/m) sum_j u_j (mean of the top j psi - psibar)``, straight from the definition."""
    ranked = psi[np.argsort(-score, kind="stable")]
    m = ranked.size
    mean = ranked.mean()
    return sum(u[j - 1] * (ranked[:j].mean() - mean) for j in range(1, m + 1)) / m


class TestCurveWeights:
    def test_autoc_weights_at_m_four(self):
        """``w_k = (H_{k,4} - 1) / 4`` with H the harmonic tail."""
        assert _curve_weights(np.ones(4)).tolist() == pytest.approx(
            [13 / 48, 1 / 48, -5 / 48, -9 / 48]
        )

    def test_qini_weights_at_m_four(self):
        """``u_j = j/m`` collapses to ``w_k = ((m+1)/2 - k) / m^2``."""
        assert _curve_weights(np.arange(1, 5) / 4.0).tolist() == pytest.approx(
            [3 / 32, 1 / 32, -1 / 32, -3 / 32]
        )

    def test_weights_sum_to_zero_at_m_four(self):
        """Exact at m=4; a score with no signal must integrate to no area."""
        assert _curve_weights(np.ones(4)).sum() == 0.0
        assert _curve_weights(np.arange(1, 5) / 4.0).sum() == 0.0

    @pytest.mark.parametrize("m", [4, 7, 13, 50, 200])
    def test_weights_sum_to_zero_at_every_size(self, m):
        # Exact in real arithmetic; float accumulation of the harmonic tail
        # leaves a residual far below the O(log m / m) weights themselves.
        assert _curve_weights(np.ones(m)).sum() == pytest.approx(0.0, abs=1e-15)
        assert _curve_weights(np.arange(1, m + 1) / m).sum() == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("m", [4, 7, 13, 50, 200])
    def test_autoc_matches_the_brute_force_double_sum(self, m):
        rng = np.random.default_rng(11)
        score, psi = rng.normal(size=m), rng.normal(size=m)
        statistic = _rank_test(score, psi, np.ones(m))
        assert statistic.estimate == pytest.approx(_brute_force_area(score, psi, np.ones(m)))

    @pytest.mark.parametrize("m", [4, 7, 13, 50, 200])
    def test_qini_matches_the_brute_force_double_sum(self, m):
        rng = np.random.default_rng(12)
        score, psi = rng.normal(size=m), rng.normal(size=m)
        u = np.arange(1, m + 1) / m
        assert _rank_test(score, psi, u).estimate == pytest.approx(_brute_force_area(score, psi, u))


class TestRankTest:
    def test_a_constant_psi_scores_exactly_zero(self):
        score = np.array([4.0, 3.0, 2.0, 1.0])
        psi = np.full(4, 2.0)
        assert _rank_test(score, psi, np.ones(4)).estimate == 0.0
        assert _rank_test(score, psi, np.arange(1, 5) / 4.0).estimate == 0.0

    def test_the_standard_error_is_the_mean_of_scores_form(self):
        rng = np.random.default_rng(4)
        score, psi = rng.normal(size=30), rng.normal(size=30)
        statistic = _rank_test(score, psi, np.ones(30))
        phi = 30 * _unit_weights(score, np.ones(30)) * psi
        assert statistic.se == pytest.approx(phi.std(ddof=1) / math.sqrt(30))

    def test_the_p_value_is_one_sided(self):
        """H1 is `estimate > 0`: a backwards ranking is never evidence of signal."""
        rng = np.random.default_rng(5)
        score, psi = rng.normal(size=60), rng.normal(size=60)
        forward = _rank_test(score, psi, np.ones(60))
        # Negating the scores negates the statistic and leaves the spread alone.
        backward = _rank_test(score, -psi, np.ones(60))
        assert forward.estimate is not None and backward.estimate is not None
        assert forward.se is not None and backward.se is not None
        assert forward.p_value is not None and backward.p_value is not None
        assert forward.estimate == pytest.approx(-backward.estimate)
        assert forward.se == pytest.approx(backward.se)
        assert forward.p_value == pytest.approx(norm.sf(forward.estimate / forward.se))
        assert forward.p_value + backward.p_value == pytest.approx(1.0)
        assert (forward.estimate > 0.0) == (forward.p_value < 0.5)

    def test_a_ranking_aligned_with_the_scores_reads_positive(self):
        rng = np.random.default_rng(8)
        psi = rng.normal(size=60)
        forward = _rank_test(psi, psi, np.ones(60))
        backward = _rank_test(-psi, psi, np.ones(60))
        assert forward.estimate is not None and backward.estimate is not None
        assert forward.estimate > 0.0
        assert backward.estimate < 0.0

    def test_a_degenerate_spread_cannot_reject(self):
        assert _rank_test(np.ones(6), np.ones(6), np.ones(6)).p_value == 1.0


def _binned_psi_vs_tau(n: int, bins: int, seed: int) -> tuple[float, float, np.ndarray]:
    """Draw ``n`` units with a linear effect on a prognostic baseline, bin by ``x``.

    Half the units are treated.  Bins are equal-width in ``x``, so binning by
    ``x`` is binning by the true ``tau``.  Returns the slope of binned
    ``mean(psi)`` regressed on binned ``tau``, the plug-in standard error of
    that slope, and the per-bin LEVEL deviations ``(mean(psi) - tau) / se``.
    Every standard error is built from the within-bin spread of ``psi`` at
    this draw count, so a caller's tolerance follows from ``n`` rather than
    from whatever happened to pass.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    tau = 0.5 + 2.0 * x
    d = np.zeros(n)
    d[rng.permutation(n)[: n // 2]] = 1.0
    y = 5.0 + 3.0 * x + d * tau + rng.standard_normal(n)
    psi = _ipw_psi(y, d)

    b = np.minimum((x * bins).astype(int), bins - 1)
    counts = np.bincount(b, minlength=bins)
    tau_b = np.bincount(b, weights=tau) / counts
    psi_b = np.bincount(b, weights=psi) / counts
    se_b = np.array([psi[b == k].std(ddof=1) for k in range(bins)]) / np.sqrt(counts)

    centered = tau_b - tau_b.mean()
    s_tt = float(centered @ centered)
    slope = float(centered @ psi_b) / s_tt
    se_slope = math.sqrt(float((centered**2) @ se_b**2)) / s_tt
    return slope, se_slope, (psi_b - tau_b) / se_b


class TestIpwPsi:
    def test_psi_is_the_centered_ipw_transform(self):
        y = np.array([1.0, 4.0, 2.0, 9.0])
        d = np.array([1.0, 0.0, 1.0, 0.0])
        expected = (y - 4.0) * (d - 0.5) / 0.25
        assert _ipw_psi(y, d).tolist() == pytest.approx(expected.tolist())

    def test_centering_leaves_the_arm_contrast_alone(self):
        """Under a balanced design the mean of psi is the difference in means."""
        y = np.array([3.0, 5.0, 11.0, 1.0, 2.0, 6.0])
        d = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        assert _ipw_psi(y, d).mean() == pytest.approx(y[d == 1].mean() - y[d == 0].mean())

    def test_a_prognostic_baseline_shrinks_the_score_spread(self):
        rng = np.random.default_rng(9)
        d = np.array([i % 2 for i in range(400)], dtype=float)
        baseline = 2.0 * rng.standard_normal(400)
        y = 10.0 + baseline + 0.5 * d
        uncentered = y * (d - 0.5) / 0.25
        assert _ipw_psi(y, d).std(ddof=1) < uncentered.std(ddof=1) / 4.0

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_the_conditional_mean_of_psi_tracks_the_true_effect(self):
        """``E[psi | x] = tau(x)`` - the property every downstream rank test rests on.

        120k units, an effect linear in ``x`` sitting on a prognostic
        baseline, half of them treated.  Bin by the true ``tau`` and the
        binned ``mean(psi)`` has to regress on the binned ``tau`` at slope
        1 AND sit at the right LEVEL: an arm-wise centering, say, keeps the
        slope and is off by ``mean(tau)`` everywhere, so the slope alone
        would not catch it.  Both tolerances are 4 plug-in standard errors
        built from the within-bin spread of ``psi`` at this draw count, so
        they follow from the simulation size rather than from what passed.
        """
        slope, se_slope, level_z = _binned_psi_vs_tau(120_000, 24, 20_240_816)

        # Guards the guard: if a future edit thins the draw out, the slope
        # tolerance below stops being a real constraint before it stops passing.
        assert se_slope < 0.03
        assert slope == pytest.approx(1.0, abs=4.0 * se_slope)
        assert np.all(np.abs(level_z) < 4.0)

    def test_the_conditional_mean_of_psi_tracks_the_true_effect_smoke(self):
        """Fast, unmarked 2k-unit variant of the 120k check above.

        Same slope-and-level property, same 4-plug-in-SE rule; only the draw
        count changes, so the tolerances widen on their own - at 2k units in
        6 bins the slope band is ~0.59 (vs ~0.05 at 120k) and the level bands
        are ~0.5-0.9 per bin.  Both mutations the big test rules out still
        fail here with room to spare: arm-wise centering biases every bin by
        ``mean(tau) = 1.5``, i.e. 14 SE off level, and dividing by ``pbar``
        instead of ``pbar (1 - pbar)`` halves the slope, 5.8 SE off.
        """
        slope, se_slope, level_z = _binned_psi_vs_tau(2_000, 6, 20_240_816)

        # Same guard as above, rescaled: 2k units give ~0.147, so the slope
        # band stops being a real constraint before this stops passing.
        assert se_slope < 0.2
        assert slope == pytest.approx(1.0, abs=4.0 * se_slope)
        assert np.all(np.abs(level_z) < 4.0)


class TestClan:
    cols = {
        "spend": np.array([1.0, 2.0, 3.0, 4.0, 10.0, 12.0]),
        "platform": np.array(["ios", "ios", "android", "ios", "android", "android"]),
    }
    most = np.array([True, True, True, False, False, False])
    least = ~most

    def test_continuous_row_is_the_two_sample_arithmetic(self):
        row = _clan(self.cols, [SPEND], self.most, self.least, Z95)[0]
        assert row.covariate == "spend"
        assert row.mean_most == pytest.approx(2.0)  # mean(1, 2, 3)
        assert row.mean_least == pytest.approx(26 / 3)  # mean(4, 10, 12)
        assert row.diff == pytest.approx(2.0 - 26 / 3)
        # var(1,2,3) = 1; var(4,10,12) = 52/3; se = sqrt(1/3 + 52/9).
        assert row.se == pytest.approx(math.sqrt(55) / 3)
        assert row.diff is not None and row.se is not None
        assert row.lb == pytest.approx(row.diff - Z95 * row.se)
        assert row.ub == pytest.approx(row.diff + Z95 * row.se)

    def test_categorical_rows_are_level_shares_including_the_modal_level(self):
        rows = _clan(self.cols, [PLATFORM], self.most, self.least, Z95)
        assert [r.covariate for r in rows] == ["platform=android", "platform=ios"]
        android, ios = rows
        assert (android.mean_most, android.mean_least) == pytest.approx((1 / 3, 2 / 3))
        assert (ios.mean_most, ios.mean_least) == pytest.approx((2 / 3, 1 / 3))
        assert ios.diff == pytest.approx(1 / 3)
        # var(0,0,1) = var(0,1,1) = 1/3, so se = sqrt(1/9 + 1/9).
        assert ios.se == pytest.approx(math.sqrt(2) / 3)

    def test_an_empty_group_yields_nan_rows(self):
        row = _clan(self.cols, [SPEND], self.most, np.zeros(6, dtype=bool), Z95)[0]
        assert row.mean_least is not None and row.diff is not None and row.se is not None
        assert math.isnan(row.mean_least) and math.isnan(row.diff) and math.isnan(row.se)


class TestValidateCateArrays:
    def test_the_split_is_honest_and_complete(self):
        v = validate_cate_arrays(**_signal_fixture(), interact=[SPEND], n_groups=4)
        assert (v.n_train, v.n_holdout) == (40, 40)
        assert sum(g.n for g in v.groups) == v.n_holdout
        assert [g.group for g in v.groups] == [1, 2, 3, 4]

    def test_the_gate_flips_with_alpha(self):
        """AUTOC p = 0.0238 on this fixture: passes at 5%, fails at 1%."""
        fixture = _signal_fixture()
        loose = validate_cate_arrays(**fixture, interact=[SPEND], n_groups=4, alpha=0.05)
        tight = validate_cate_arrays(**fixture, interact=[SPEND], n_groups=4, alpha=0.01)
        assert loose.autoc.p_value == pytest.approx(tight.autoc.p_value)
        assert loose.autoc.p_value == pytest.approx(0.02383, abs=5e-5)
        assert loose.passed and not tight.passed
        assert (loose.alpha, tight.alpha) == (0.05, 0.01)

    def test_the_holdout_ate_is_the_holdout_difference_in_means(self):
        fixture = _signal_fixture()
        v = validate_cate_arrays(**fixture, interact=[SPEND], n_groups=4)
        hold = _content_hash_holdout(IDS_80)
        y, d = fixture["y"][hold], fixture["d"][hold]
        assert v.holdout_ate is not None
        assert v.holdout_ate.value == pytest.approx(y[d == 1].mean() - y[d == 0].mean())
        assert v.holdout_ate.level == 0.95

    def test_group_intervals_are_bonferroni_corrected_across_n_groups(self):
        """Five group intervals cut at the unadjusted z face
        ~1-(1-alpha)^5 familywise error; CDDF (2018) Bonferroni-corrects
        across the reported family instead."""
        fixture = _signal_fixture()
        n_groups = 4
        v = validate_cate_arrays(**fixture, interact=[SPEND], n_groups=n_groups, alpha=0.05)
        expected_z = float(norm.ppf(1.0 - 0.05 / (2.0 * n_groups)))
        assert expected_z > Z95  # the correction must actually widen the interval
        for g in v.groups:
            assert g.se is not None and g.effect is not None
            assert g.lb is not None and g.ub is not None
            if not math.isnan(g.se):
                assert (g.ub - g.effect) == pytest.approx(expected_z * g.se, rel=1e-10)
                assert (g.effect - g.lb) == pytest.approx(expected_z * g.se, rel=1e-10)

    def test_clan_intervals_are_bonferroni_corrected_across_row_count(self):
        """CLAN grows with covariate/level count, same familywise risk
        as GATES; Bonferroni-correct across every row CLAN actually reports."""
        fixture = _signal_fixture()
        v = validate_cate_arrays(
            **fixture, interact=[SPEND], adjust=[PLATFORM], n_groups=4, alpha=0.05
        )
        n_rows = len(v.clan)
        assert n_rows == 3  # spend, platform=android, platform=ios
        expected_z = float(norm.ppf(1.0 - 0.05 / (2.0 * n_rows)))
        assert expected_z > Z95
        for r in v.clan:
            assert r.se is not None and r.diff is not None and r.ub is not None
            if not math.isnan(r.se):
                assert (r.ub - r.diff) == pytest.approx(expected_z * r.se, rel=1e-10)

    def test_split_caveat_names_the_single_split(self):
        """Nothing on CateValidation used to disclose that every
        number is read off one deterministic split, not CDDF's medians
        over many; this is a caveat, not a fix -- multi-split is out of
        scope here."""
        v = validate_cate_arrays(**_signal_fixture(), interact=[SPEND], n_groups=4)
        assert "split" in v.split_caveat.lower()

    def test_clan_profiles_the_interaction_and_adjustment_sets(self):
        v = validate_cate_arrays(
            **_signal_fixture(), interact=[SPEND], adjust=[PLATFORM], n_groups=4
        )
        assert [r.covariate for r in v.clan] == ["spend", "platform=android", "platform=ios"]
        # The score is increasing in spend, so the top group must be too.
        assert v.clan[0].diff is not None
        assert v.clan[0].diff > 0.0

    def test_the_rank_tests_agree_with_the_definition_on_the_holdout(self):
        fixture = _signal_fixture()
        v = validate_cate_arrays(**fixture, interact=[SPEND], n_groups=4)
        score, psi = _refit_holdout(fixture, [SPEND])
        m = score.size
        assert v.autoc.estimate == pytest.approx(_brute_force_area(score, psi, np.ones(m)))
        assert v.qini.estimate == pytest.approx(
            _brute_force_area(score, psi, np.arange(1, m + 1) / m)
        )

    def test_a_score_with_no_surviving_interaction_is_flat_and_scores_zero(self):
        """Everything pruned means one constant score: no ranking, no area."""
        fixture = _tied_fixture()
        y = 1.0 + fixture["d"] * 0.5 + np.array([((i * 37) % 11 - 5) / 5 for i in range(40)])
        v = validate_cate_arrays(
            y,
            fixture["d"],
            fixture["cols"],
            IDS_40,
            interact=[PLATFORM],
            n_groups=2,
        )
        assert v.autoc.estimate == pytest.approx(0.0, abs=1e-12)
        assert v.qini.estimate == pytest.approx(0.0, abs=1e-12)
        assert not v.passed

    def test_a_mismatched_column_length_is_refused_by_name(self):
        fixture = _signal_fixture()
        cols = dict(fixture["cols"], spend=np.linspace(-2.0, 2.0, 79))
        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate_arrays(fixture["y"], fixture["d"], cols, IDS_80, interact=[SPEND])
        assert exc_info.value.code == "estimation.targeting.covariate_rows_expected"

    def test_a_non_binary_treatment_is_refused(self):
        fixture = _signal_fixture()
        d = fixture["d"].copy()
        d[0] = 2.0
        with pytest.raises(InvalidRequestError) as exc_info:
            validate_cate_arrays(fixture["y"], d, fixture["cols"], IDS_80, interact=[SPEND])
        assert exc_info.value.code == "estimation.cate.treatment_binary"


class TestTargetingRule:
    """The deployable rule, and the refusal that is its default answer."""

    def test_a_passing_gate_populates_every_policy_field(self):
        rule = targeting_rule_arrays(**_step_fixture(), interact=[SPEND], n_groups=4, fraction=0.4)
        assert rule.recommendation == "target"
        assert rule.fraction == 0.4
        assert rule.validation.passed
        assert rule.threshold is not None
        assert rule.policy_value is not None
        assert rule.uplift_vs_average is not None

    def test_public_cate_results_have_concise_meaningful_representations(self):
        rule = targeting_rule_arrays(**_step_fixture(), interact=[SPEND], n_groups=4, fraction=0.4)
        validation = rule.validation

        validation_display = repr(validation)
        rule_display = repr(rule)
        group_display = repr(validation.groups[-1])
        clan_display = repr(validation.clan[0])

        assert len(validation_display) < 500
        assert "gate=passed" in validation_display
        assert "AUTOC" in validation_display
        assert "n_holdout" in validation_display
        assert "evaluation_population" not in validation_display
        assert len(rule_display) < 500
        assert "recommendation='target'" in rule_display
        assert "fraction=0.4" in rule_display
        assert "threshold=" in rule_display
        assert "policy_value=+2.5" in rule_display and "point only" in rule_display
        assert "validation=" not in rule_display
        assert str(rule) == rule_display
        from rich.pretty import pretty_repr

        rich_display = pretty_repr(validation)
        assert "gate" in rich_display and "Bonferroni-corrected" in rich_display
        assert "evaluation_population" not in rich_display
        assert "effect=" in group_display and "interval=" in group_display
        assert "support_failures" not in group_display
        assert "difference=" in clan_display and "interval=" in clan_display
        assert "bootstrap_seed" not in clan_display

    def test_cate_rank_repr_keeps_unavailable_reasons(self):
        rule = targeting_rule_arrays(**_step_fixture(), interact=[SPEND], n_groups=4, fraction=0.4)
        rank = rule.validation.autoc.model_copy(
            update={"p_value": None, "unavailable_reason": "rank_test_unavailable"}
        )
        validation = rule.validation.model_copy(update={"autoc": rank, "passed": False})
        display = repr(validation)
        assert "AUTOC=+0.7255, p=unavailable" in display
        assert "unavailable_reason='rank_test_unavailable'" in display

    def test_targeting_repr_discloses_target_population_and_weighting(self):
        rule = targeting_rule_arrays(**_step_fixture(), interact=[SPEND], n_groups=4, fraction=0.4)
        display = repr(rule)
        assert "population='full target population'" in display
        assert "weighting='member_count'" in display
        from rich.pretty import pretty_repr

        rich_display = pretty_repr(rule)
        assert "population" in rich_display and "weighting" in rich_display

    def test_failed_gate_repr_preserves_unavailability_reason(self):
        rule = targeting_rule_arrays(
            **_step_fixture(), interact=[SPEND], n_groups=4, fraction=0.4, alpha=0.01
        )
        assert not rule.validation.passed
        validation_display = repr(rule.validation)
        assert "gate=failed" in validation_display
        assert "AUTOC" in validation_display
        assert "evaluation_population" not in validation_display
        from rich.pretty import pretty_repr

        rich_display = pretty_repr(rule)
        assert "recommendation" in rich_display and "policy_intervals" in rich_display
        assert "evaluation_population" not in rich_display
        display = repr(rule)
        assert "recommendation='simple'" in display
        assert "AUTOC" in display
        assert "threshold=unavailable" in display
        assert "vs alpha=" in display
        assert "validation=" not in display

    def test_a_small_valid_fraction_reports_a_finite_point_value(self):
        """policy_value/uplift_vs_average ship without an interval
        (post-selection: the same holdout both gates and reports), but the
        point value itself is still a finite number."""
        rule = targeting_rule_arrays(**_step_fixture(), interact=[SPEND], n_groups=4, fraction=0.1)
        assert rule.recommendation == "target"
        assert rule.policy_value is not None
        assert math.isfinite(rule.policy_value.value)
        assert rule.policy_value.lb is None and rule.policy_value.ub is None

    def test_a_too_small_fraction_is_refused_before_welch(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            targeting_rule_arrays(**_step_fixture(), interact=[SPEND], n_groups=4, fraction=0.05)
        assert exc_info.value.code == "estimation.targeting.targeting_fraction_leaves"

    def test_the_policy_value_is_the_top_fraction_difference_in_means(self):
        """Noiseless by construction: 3.5 treated against 1.0 control, exactly.

        vdb5: no interval ships for this post-selection value (the gate and
        this report share the same holdout), matching the winner's-curse
        handling already used elsewhere in this module."""
        rule = targeting_rule_arrays(**_step_fixture(), interact=[SPEND], n_groups=4, fraction=0.4)
        assert rule.policy_value is not None
        assert rule.policy_value.value == 2.5
        assert (rule.policy_value.lb, rule.policy_value.ub) == (None, None)
        assert rule.policy_value.level is None

    def test_the_threshold_is_the_holdout_score_quantile(self):
        fixture = _signal_fixture()
        rule = targeting_rule_arrays(**fixture, interact=[SPEND], n_groups=4, fraction=0.3)
        score, _ = _refit_holdout(fixture, [SPEND])
        assert rule.threshold == pytest.approx(float(np.quantile(score, 0.7)))
        # The cut IS the deployable rule, so it has to select the share asked for.
        assert int((score >= rule.threshold).sum()) == 12  # 30% of a 40-unit holdout

    def test_the_uplift_is_the_targeting_curve_at_the_cut(self):
        """No interval ships for this post-selection value."""
        fixture = _signal_fixture()
        rule = targeting_rule_arrays(**fixture, interact=[SPEND], n_groups=4, fraction=0.3)
        score, psi = _refit_holdout(fixture, [SPEND])
        assert rule.uplift_vs_average is not None
        # A spike of height m at rank 12 reads the curve at 12/40 and nowhere else.
        u = np.zeros(40)
        u[11] = 40.0
        assert rule.uplift_vs_average.value == pytest.approx(_brute_force_area(score, psi, u))
        assert (rule.uplift_vs_average.lb, rule.uplift_vs_average.ub) == (None, None)

    def test_a_failed_gate_refuses_with_the_validation_still_attached(self):
        """The refusal is a RESULT: the caller sees exactly why it was refused."""
        rule = targeting_rule_arrays(
            **_step_fixture(), interact=[SPEND], n_groups=4, fraction=0.4, alpha=0.01
        )
        assert rule.recommendation == "simple"
        assert rule.threshold is None
        assert rule.policy_value is None
        assert rule.uplift_vs_average is None
        # Same fraction, same evidence - only the verdict changed.
        assert rule.fraction == 0.4
        assert not rule.validation.passed
        assert rule.validation.autoc.p_value == pytest.approx(0.02540, abs=5e-5)
        assert [g.group for g in rule.validation.groups] == [1, 2, 3, 4]
        assert rule.validation.holdout_ate is not None
        assert not math.isnan(rule.validation.holdout_ate.value)

    def test_the_attached_validation_is_the_gate_verbatim(self):
        fixture = _step_fixture()
        rule = targeting_rule_arrays(**fixture, interact=[SPEND], n_groups=4, fraction=0.4)
        assert rule.validation == validate_cate_arrays(**fixture, interact=[SPEND], n_groups=4)

    @pytest.mark.parametrize("fraction", [-0.1, 1.5, math.nan])
    def test_a_fraction_outside_the_unit_interval_is_refused(self, fraction):
        with pytest.raises(InvalidRequestError) as exc_info:
            targeting_rule_arrays(**_step_fixture(), interact=[SPEND], fraction=fraction)
        assert exc_info.value.code == "estimation.targeting.fraction_share_units"

    def test_fraction_has_no_default(self):
        """Pre-commitment is the feature; a default would quietly undo it."""
        with pytest.raises(TypeError):
            targeting_rule_arrays(**_step_fixture(), interact=[SPEND])
        # The keyword is still spelled `fraction` and still taken explicitly.
        assert targeting_rule_arrays(
            **_step_fixture(), interact=[SPEND], fraction=0.4
        ).fraction == pytest.approx(0.4)


def _selection_fixture(n: int = 200) -> dict:
    """NOISELESS step CATE: the top 20% by spend has effect 2.5, the rest 0.5.

    With cost_per_treated=1.0 the net benefit f*(E[tau | top f] - cost) is
    maximized exactly at fraction 0.2: below it every marginal unit adds
    2.5 - 1.0, above it every marginal unit COSTS 1.0 - 0.5.
    """
    spend = np.linspace(-2.0, 2.0, n)
    d = np.array([i % 2 for i in range(n)], dtype=float)
    y = 1.0 + 0.3 * spend + d * (0.5 + 2.0 * (spend > 1.2))
    return {
        "y": y,
        "d": d,
        "cols": {"spend": spend},
        "unit_ids": np.array([f"u{i}" for i in range(n)]),
    }


class TestSelectTargetingRuleArrays:
    @staticmethod
    def _selection_with_different_inner_outer_retention(*, include_evaluation_population=False):
        fixture = _selection_fixture()
        scored_populations: list[tuple[str, ...]] = []

        def trim_inner_only(y, d, X, unit_ids, cluster_ids):
            del d, X, cluster_ids
            scored_populations.append(tuple(unit_ids))
            if len(scored_populations) == 1:
                kept = np.fromiter((not unit_id.endswith("0") for unit_id in unit_ids), dtype=bool)
            else:
                kept = np.ones(unit_ids.shape, dtype=bool)
            return np.asarray(y, dtype=float), kept

        result = select_targeting_rule_arrays(
            **fixture,
            interact=[SPEND],
            fractions=(0.1, 0.2),
            n_folds=4,
            seed=11,
            psi_fn=trim_inner_only,
            arm_summary="score",
            include_evaluation_population=include_evaluation_population,
        )
        return result, scored_populations

    def test_observational_selection_uses_dr_scores_for_the_inner_objective(self):
        rng = np.random.default_rng(0)
        n = 1_200
        spend = rng.normal(size=n)
        d = rng.binomial(1, expit(0.6 * spend)).astype(float)
        y = -6.0 * spend + d * (1.0 + 1.5 * spend) + rng.normal(0.0, 0.5, n)
        psi_fn = functools.partial(
            _dr_psi,
            propensity_learner=LogisticPropensity,
            outcome_learner=RidgeOutcome,
            folds=5,
            seed=0,
            gate=IdentificationGate(overlap="trim"),
        )

        unit_ids = np.array([f"u{i}" for i in range(n)])
        ipw = select_targeting_rule_arrays(
            y,
            d,
            {"spend": spend},
            unit_ids,
            interact=[SPEND],
            fractions=(0.1, 0.2, 0.4, 0.8),
            n_folds=4,
            seed=11,
        )
        dr = select_targeting_rule_arrays(
            y,
            d,
            {"spend": spend},
            unit_ids,
            interact=[SPEND],
            fractions=(0.1, 0.2, 0.4, 0.8),
            n_folds=4,
            seed=11,
            adjustment=("spend",),
            psi_fn=psi_fn,
            arm_summary="score",
        )

        assert ipw.selected_fraction == 0.4
        assert dr.selected_fraction == 0.8

    def test_inner_overlap_mask_trims_every_array_before_fold_scoring(self):
        fixture = _selection_fixture()
        scored_populations: list[tuple[str, ...]] = []

        def trim_score(y, d, X, unit_ids, cluster_ids):
            del d, X, cluster_ids
            scored_populations.append(tuple(unit_ids))
            kept = np.fromiter((not unit_id.endswith("0") for unit_id in unit_ids), dtype=bool)
            return np.asarray(y, dtype=float), kept

        result = select_targeting_rule_arrays(
            **fixture,
            interact=[SPEND],
            fractions=(0.1, 0.2),
            n_folds=4,
            seed=11,
            psi_fn=trim_score,
            arm_summary="score",
        )

        assert len(scored_populations) == 2
        assert len(scored_populations[0]) == 100
        expected_inner_n = sum(not unit_id.endswith("0") for unit_id in scored_populations[0])
        assert {row.n for row in result.inner} == {expected_inner_n}

    def test_inner_overlap_population_trains_the_final_outer_rule(self):
        result, scored_populations = self._selection_with_different_inner_outer_retention()

        assert len(scored_populations) == 2
        retained_inner_n = sum(not unit_id.endswith("0") for unit_id in scored_populations[0])
        assert result.rule.validation.n_train == retained_inner_n

    def test_inner_population_serializes_independently_from_the_outer_rule(self):
        result, scored_populations = self._selection_with_different_inner_outer_retention()

        assert len(scored_populations) == 2
        assert len(scored_populations[0]) == len(scored_populations[1])
        payload = result.model_dump(mode="json")
        assert payload["population"] == "overlap_subpopulation"
        assert payload["rule"]["population"] is None
        assert TargetingSelection.model_validate_json(result.model_dump_json()) == result

    def test_selection_lands_on_the_planted_optimal_fraction(self):
        result = select_targeting_rule_arrays(
            **_selection_fixture(),
            interact=[SPEND],
            fractions=(0.1, 0.2, 0.4),
            cost_per_treated=1.0,
            n_folds=4,
            seed=11,
        )
        assert isinstance(result, TargetingSelection)
        assert result.selected_fraction == 0.2
        assert result.rule.fraction == 0.2
        assert result.fractions == (0.1, 0.2, 0.4)
        assert [s.fraction for s in result.inner] == [0.1, 0.2, 0.4]

    def test_the_selected_rows_net_benefit_ships_without_an_interval(self):
        """The argmax row is winner's-curse inflated -- picking the
        biggest of several noisy estimates -- but used to be shape-identical
        to the honest, non-selected rows. Only the selected row omits its
        interval; the others keep theirs."""
        result = select_targeting_rule_arrays(
            **_selection_fixture(),
            interact=[SPEND],
            fractions=(0.1, 0.2, 0.4),
            cost_per_treated=1.0,
            n_folds=4,
            seed=11,
        )
        assert result.selected_fraction == 0.2
        selected = next(s for s in result.inner if s.fraction == result.selected_fraction)
        others = [s for s in result.inner if s.fraction != result.selected_fraction]
        assert others  # the fixture's grid has more than one fraction
        assert selected.net_benefit.lb is None
        assert selected.net_benefit.ub is None
        assert selected.net_benefit.level is None
        for row in others:
            assert row.net_benefit.lb is not None
            assert row.net_benefit.ub is not None
            assert row.net_benefit.level == 0.95

    def test_a_singleton_grid_keeps_its_interval(self):
        result = select_targeting_rule_arrays(
            **_selection_fixture(),
            interact=[SPEND],
            fractions=(0.2,),
            cost_per_treated=1.0,
            n_folds=4,
            seed=11,
        )
        estimate = result.inner[0].net_benefit
        assert estimate.lb is not None
        assert estimate.ub is not None
        assert estimate.level == 0.95

    def test_outer_validation_names_the_seeded_stratified_split(self):
        result = select_targeting_rule_arrays(
            **_selection_fixture(), interact=[SPEND], fractions=(0.2,), n_folds=4, seed=11
        )
        caveat = result.rule.validation.split_caveat.lower()
        assert "seeded" in caveat
        assert "stratified" in caveat
        assert "crc32" not in caveat

    def test_zero_cost_degenerates_to_total_benefit(self):
        # With no cost, adding units with a positive effect always helps:
        # the widest fraction wins.
        result = select_targeting_rule_arrays(
            **_selection_fixture(),
            interact=[SPEND],
            fractions=(0.1, 0.2, 0.4),
            n_folds=4,
            seed=11,
        )
        assert result.selected_fraction == 0.4
        assert result.cost_per_treated == 0.0

    def test_the_outer_rule_reuses_the_existing_machinery(self):
        result = select_targeting_rule_arrays(
            **_selection_fixture(),
            interact=[SPEND],
            fractions=(0.1, 0.2, 0.4),
            cost_per_treated=1.0,
            n_folds=4,
            seed=11,
        )
        rule = result.rule
        assert rule.recommendation == "target"
        assert rule.threshold is not None
        # Locked at 0.2, the targeted outer units are all in the high step:
        # treated read 3.5-ish, control 1.0-ish, so the value is near 2.5.
        assert rule.policy_value is not None
        assert rule.policy_value.value == pytest.approx(2.5, abs=0.3)

    def test_selection_never_touches_the_gate(self):
        # cost=1.5 -> 0.1 wins (net 0.10 vs 0.0); cost=0 -> 0.4 wins (net 0.60
        # vs 0.25) - margins wide enough that validation matches regardless.
        fixture = _selection_fixture()
        cheap = select_targeting_rule_arrays(
            **fixture, interact=[SPEND], fractions=(0.1, 0.4), n_folds=4, seed=11
        )
        costly = select_targeting_rule_arrays(
            **fixture,
            interact=[SPEND],
            fractions=(0.1, 0.4),
            cost_per_treated=1.5,
            n_folds=4,
            seed=11,
        )
        assert cheap.selected_fraction == 0.4
        assert costly.selected_fraction == 0.1
        assert cheap.rule.validation == costly.rule.validation

    def test_a_failed_gate_refuses_policy_numbers_but_keeps_the_inner_table(self):
        rng = np.random.default_rng(3)
        n = 200
        fixture = {
            "y": rng.standard_normal(n),  # pure noise: no heterogeneity, no effect
            "d": np.array([i % 2 for i in range(n)], dtype=float),
            "cols": {"spend": np.linspace(-2.0, 2.0, n)},
            "unit_ids": np.array([f"u{i}" for i in range(n)]),
        }
        result = select_targeting_rule_arrays(
            **fixture,  # ty: ignore[invalid-argument-type]
            interact=[SPEND],
            fractions=(0.1, 0.2),
            n_folds=4,
            seed=11,
        )
        assert result.rule.recommendation == "simple"
        assert result.rule.policy_value is None
        assert len(result.inner) == 2

    def test_deterministic_given_seed(self):
        kwargs = dict(
            **_selection_fixture(),
            interact=[SPEND],
            fractions=(0.1, 0.2, 0.4),
            cost_per_treated=1.0,
            n_folds=4,
        )
        first = select_targeting_rule_arrays(**kwargs, seed=11)  # ty: ignore[invalid-argument-type]
        second = select_targeting_rule_arrays(**kwargs, seed=11)  # ty: ignore[invalid-argument-type]
        assert first == second

    def test_an_empty_grid_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            select_targeting_rule_arrays(
                **_selection_fixture(), interact=[SPEND], fractions=(), seed=11
            )
        assert exc_info.value.code == "estimation.targeting.fractions_non_empty"

    def test_an_out_of_range_fraction_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            select_targeting_rule_arrays(
                **_selection_fixture(), interact=[SPEND], fractions=(0.2, 1.1), seed=11
            )
        assert exc_info.value.code == "estimation.targeting.every_fraction"

    def test_duplicate_fractions_are_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            select_targeting_rule_arrays(
                **_selection_fixture(), interact=[SPEND], fractions=(0.2, 0.2), seed=11
            )
        assert exc_info.value.code == "estimation.targeting.fractions_contains_duplicates"

    @pytest.mark.parametrize(
        "n,code",
        [
            (12, "estimation.crossfit.insufficient_clusters"),
            (16, "estimation.targeting.validation_fold_holds"),
        ],
    )
    def test_an_arm_starved_fold_is_refused_by_name(self, n, code):
        # The inner half needs one unit per arm per fold for assignment,
        # and two per arm per validation fold for targeting's summaries.
        tiny = {
            "y": np.arange(n, dtype=float),
            "d": np.array([i % 2 for i in range(n)], dtype=float),
            "cols": {"spend": np.arange(n, dtype=float)},
            "unit_ids": np.array([f"u{i}" for i in range(n)]),
        }
        with pytest.raises(InvalidRequestError) as exc_info:
            select_targeting_rule_arrays(
                **tiny,  # ty: ignore[invalid-argument-type]
                interact=[SPEND],
                fractions=(0.2,),
                n_folds=4,
                seed=11,
            )
        assert exc_info.value.code == code


# Independent weighted targets and cluster-level uncertainty oracles.
def test_cluster_validation_member_and_equal_targets_differ_on_informative_sizes():
    from increment.estimation.targeting import _cluster_mean, _target_weights

    ids = np.array(["small", "large", "large", "large", "medium", "medium"])
    scores = np.array([1.0, 9.0, 9.0, 9.0, 4.0, 4.0])
    member = _cluster_mean(scores, _target_weights(ids, "member_count"), ids)
    equal = _cluster_mean(scores, _target_weights(ids, "equal"), ids)
    assert member.value == pytest.approx(36 / 6)
    assert equal.value == pytest.approx(14 / 3)
    assert member.value != pytest.approx(equal.value)
    assert member.n_clusters == equal.n_clusters == 3
    # Equal weighting estimates a mean of cluster means, with cluster-only df.
    assert equal.se is not None
    assert equal.se**2 == pytest.approx(sum((v - 14 / 3) ** 2 for v in (1, 9, 4)) / 6)
    assert equal.df == 2


def test_cluster_clan_overlap_combines_signed_contributions_before_squaring():
    from increment.estimation.targeting import _clan_row

    ids = np.array(["a", "a", "b", "b", "c", "c"])
    x = np.array([2.0, 1.0, 6.0, 5.0, 13.0, 9.0])
    most = np.array([True, False] * 3)
    least = ~most
    result = _clan_row("x", x, most, least, Z95, weights=np.ones(6), cluster_ids=ids)
    # Cluster contributions to mean(high)-mean(low): (-1,-1,2)/3.
    expected_variance = (3 / 2) * ((-1 / 3) ** 2 + (-1 / 3) ** 2 + (2 / 3) ** 2)
    independent_variance = (np.var([2, 6, 13], ddof=1) + np.var([1, 5, 9], ddof=1)) / 3
    assert result.diff == pytest.approx(2.0)
    assert result.se is not None
    assert result.se**2 == pytest.approx(expected_variance)
    assert result.se**2 != pytest.approx(independent_variance)
    assert result.reference_df == 2
    assert result.uncertainty_method == "overlap-signed-combined"


def test_cluster_disjoint_arm_welch_uses_each_arms_own_scale_and_degrees():
    from increment.estimation.targeting import _cluster_contrast

    a = np.array([2.0, 7.0, 7.0, 10.0, 10.0, 10.0])
    b = np.array([0.0, 0.0, 3.0, 8.0])
    ga = np.array(["a", "b", "b", "c", "c", "c"])
    gb = np.array(["d", "d", "e", "f"])
    result = _cluster_contrast(a, b, np.ones(6), np.ones(4), ga, gb)
    ma, mb = 46 / 6, 11 / 4
    ua = np.array([2 - ma, 2 * (7 - ma), 3 * (10 - ma)]) / 6
    ub = np.array([2 * (0 - mb), 3 - mb, 8 - mb]) / 4
    va, vb = 3 / 2 * (ua @ ua), 3 / 2 * (ub @ ub)
    assert result.value == pytest.approx(ma - mb)
    assert result.se is not None
    assert result.se**2 == pytest.approx(va + vb)
    assert result.df == pytest.approx((va + vb) ** 2 / (va**2 / 2 + vb**2 / 2))
    assert result.method == "disjoint-arm-Welch"
    assert result.n_clusters == 6


def test_cluster_pooled_dr_influence_does_not_partition_mixed_arm_clusters():
    from increment.estimation.targeting import _cluster_effect

    ids = np.repeat(["a", "b", "c"], 2)
    d = np.tile([0.0, 1.0], 3)
    psi = np.array([1.0, 3.0, 5.0, 7.0, 8.0, 12.0])
    result = _cluster_effect(psi * 10, d, psi, np.ones(6), ids, "score")
    # Pooled mean=6; each cluster's centered contribution is (-8,0,8)/6.
    assert result.value == 6.0
    assert result.se is not None
    assert result.se**2 == pytest.approx(3 / 2 * (128 / 36))
    assert result.df == 2
    assert result.method == "pooled-DR-influence"


def test_cluster_tied_ranking_and_groups_are_deterministic_under_row_permutation():
    from increment.estimation.targeting import _target_weights, _weighted_bins, _weighted_ranks

    score = np.array([1.0, 0.0, 1.0, 2.0, 0.0, 2.0])
    psi = np.array([2.0, 4.0, 6.0, 8.0, 3.0, 7.0])
    ids = np.array(["a", "a", "b", "b", "b", "c"])
    weights = _target_weights(ids, "equal")
    permutation = np.array([4, 2, 5, 0, 3, 1])
    original = _weighted_bins(score, weights, 3)
    reordered = _weighted_bins(score[permutation], weights[permutation], 3)
    np.testing.assert_array_equal(reordered, original[permutation])
    assert _weighted_ranks(score, psi, weights) == pytest.approx(
        _weighted_ranks(score[permutation], psi[permutation], weights[permutation])
    )
    assert original[0] == original[2]
    assert original[1] == original[4]


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_cluster_row_cloning_preserves_targets_covariance_and_rank_integrals(weighting):
    from increment.estimation.targeting import (
        _cluster_contrast,
        _cluster_mean,
        _target_weights,
        _weighted_bins,
        _weighted_ranks,
    )

    ids = np.array(["a", "a", "b", "b", "b", "c", "c", "c", "c"])
    psi = np.array([1.0, 2.0, 5.0, 3.0, 8.0, 4.0, 5.0, 10.0, 9.0])
    score = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.0])
    weights = _target_weights(ids, weighting)
    cloned_ids, cloned_psi, cloned_score = (np.tile(x, 3) for x in (ids, psi, score))
    cloned_weights = _target_weights(cloned_ids, weighting)
    original = _cluster_mean(psi, weights, ids)
    cloned = _cluster_mean(cloned_psi, cloned_weights, cloned_ids)
    assert cloned.value == pytest.approx(original.value)
    assert cloned.se == pytest.approx(original.se)
    assert cloned.df == original.df == 2
    most, least = score > 0, score < 2
    args = (psi[most], psi[least], weights[most], weights[least], ids[most], ids[least])
    overlap = _cluster_contrast(*args)
    cloned_args = tuple(np.tile(v, 3) for v in args)
    cloned_overlap = _cluster_contrast(*cloned_args)
    assert cloned_overlap.value == pytest.approx(overlap.value)
    assert cloned_overlap.se == pytest.approx(overlap.se)
    assert _weighted_ranks(cloned_score, cloned_psi, cloned_weights) == pytest.approx(
        _weighted_ranks(score, psi, weights)
    )
    np.testing.assert_array_equal(
        _weighted_bins(cloned_score, cloned_weights, 2),
        np.tile(_weighted_bins(score, weights, 2), 3),
    )


def test_cluster_unavailable_uncertainty_distinguishes_empty_single_and_zero_variance():
    from increment.estimation.targeting import _cluster_mean

    single = _cluster_mean(np.array([1.0, 3.0]), np.ones(2), np.array(["a", "a"]))
    empty = _cluster_mean(np.array([]), np.array([]), np.array([], dtype=str))
    zero = _cluster_mean(np.zeros(3), np.ones(3), np.array(["a", "b", "c"]))
    assert single.value == 2 and single.se is None and single.df is None
    assert single.reason == "estimation.targeting.insufficient_clusters"
    assert empty.value is None and empty.se is None
    assert empty.reason == "estimation.targeting.empty_group"
    assert zero.value == 0 and zero.se is None
    assert zero.reason == "estimation.targeting.degenerate_cluster_variance"


def test_cluster_bootstrap_centering_and_interval_share_the_same_null_distribution():
    from increment.estimation.targeting import _centered_bootstrap

    # Roots are -2, -1, 1/2, 1, each repeated five times; alpha/2 selects
    # the second and nineteenth order statistics, with original scale two.
    samples = [0.0, 1.0, 3.0, 6.0] * 5
    scales = [1.0, 1.0, 2.0, 4.0] * 5
    result = _centered_bootstrap(2.0, samples, 0.2, scale=2.0, scales=scales)
    assert result.p_value == pytest.approx(6 / 21)
    assert result.se == pytest.approx(math.sqrt(105 / 19))
    assert (result.lb, result.ub) == pytest.approx((0.0, 6.0))
    # The old mean-centered, unstudentized interval was (-1.5, 4.5).
    shifted = _centered_bootstrap(
        1002.0, [v + 1000 for v in samples], 0.2, scale=2.0, scales=scales
    )
    assert shifted.se == result.se
    assert (shifted.lb, shifted.ub) == pytest.approx((1000.0, 1006.0))
    biased = _centered_bootstrap(2.0, [v + 1 for v in samples], 0.2, scale=2.0, scales=scales)
    assert (biased.lb, biased.ub) != (result.lb, result.ub)
    degenerate = _centered_bootstrap(0.0, [3.0] * 20, 0.2, scale=1.0, scales=[1.0] * 20)
    assert degenerate.se is None and degenerate.p_value is None
    assert degenerate.lb is None and degenerate.ub is None
    assert degenerate.reason == "estimation.targeting.bootstrap_zero_variance"


def test_cluster_bootstrap_tail_allocation_uses_exact_family_budget():
    from fractions import Fraction

    from increment.estimation.targeting import _centered_bootstrap

    # Binary64 0.3 is below 3/10. Rounded 20*0.3/2 is 3, but the
    # conservative order is floor(20*Fraction(0.3)/2) == 2.
    values = [float(i) for i in range(-9, 10)]
    result = _centered_bootstrap(0.0, values, 0.3, scale=1.0, scales=[1.0] * 19)
    assert (result.lb, result.ub) == (-8.0, 8.0)
    family = _centered_bootstrap(0.0, values, 0.3, scale=1.0, scales=[1.0] * 19, family=2)
    assert (family.lb, family.ub) == (-9.0, 9.0)
    assert result.p_value is not None
    assert Fraction(result.p_value) >= Fraction(11, 20)
    unresolved = _centered_bootstrap(
        0.0,
        values,
        float.fromhex("0x0.0000000000001p-1022"),
        scale=1.0,
        scales=[1.0] * 19,
        family=2,
    )
    assert unresolved.reason == "estimation.targeting.bootstrap_tail_resolution"


@pytest.mark.parametrize("bad_scale", [None, 0.0, -1.0, math.inf, math.nan])
def test_cluster_bootstrap_does_not_discard_unavailable_studentizing_scale(bad_scale):
    from increment.estimation.targeting import _centered_bootstrap

    result = _centered_bootstrap(
        2.0,
        [0.0, 1.0, 3.0, 6.0] * 5,
        0.2,
        scale=2.0,
        scales=[bad_scale] + [1.0] * 19,
    )
    assert result.valid == 20
    assert result.reason is None
    assert result.p_value == pytest.approx(11 / 21)
    assert (result.lb, result.ub) == pytest.approx((-6.0, 6.0))


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_cluster_gates_bootstrap_t_matches_exact_ratio_draw_oracle(weighting, monkeypatch):
    from fractions import Fraction

    from scipy.stats import t

    from increment.estimation import targeting

    masses = (1, 1, 1, 9)
    response = (0, 1, 2, 8)
    ids = np.repeat(np.arange(4), [2 * m for m in masses])
    score = np.concatenate([np.tile([-1.0, 1.0], m) for m in masses])
    y = np.repeat(response, [2 * m for m in masses]) + (score > 0)
    holdout = targeting._Holdout(
        n_train=y.size,
        score=score,
        y=y,
        d=np.tile([0.0, 1.0], sum(masses)),
        unit_ids=np.arange(ids.size).astype(str),
        cluster_ids=ids,
        cols={},
        covariates=(),
        cluster_weight=weighting,
    )
    draws = [(0, 0, 1, 2), (0, 1, 2, 3), (1, 2, 3, 3), (0, 2, 2, 3)] * 5
    chosen = iter(draws)

    def draw(population, rng):
        return np.array(next(chosen))

    def oracle(labels):
        w = [masses[g] if weighting == "member_count" else 1 for g in labels]
        mean = Fraction(sum(m * response[g] for m, g in zip(w, labels, strict=True)), sum(w))
        u = [m * (response[g] - mean) / sum(w) for m, g in zip(w, labels, strict=True)]
        variance = Fraction(len(labels), len(labels) - 1) * sum(v * v for v in u)
        return float(mean), math.sqrt(float(variance))

    point, scale = oracle((0, 1, 2, 3))
    samples = [oracle(labels) for labels in draws]
    roots = [(mean - point) / se for mean, se in samples]
    expected = (point - scale * max(roots), point - scale * min(roots))
    deleted = [oracle(tuple(g for g in range(4) if g != omitted))[0] for omitted in range(4)]
    jackknife_se = math.sqrt(3 / 4 * sum((mean - point) ** 2 for mean in deleted))
    half = t.isf(0.1, 3) * jackknife_se
    expected = (min(expected[0], point - half), max(expected[1], point + half))
    monkeypatch.setattr(targeting, "_cluster_resample", draw)
    result = targeting._validation(
        holdout, y, n_groups=2, alpha=0.4, arm_summary="score", bootstrap_repetitions=20
    )
    for shift, group in enumerate(result.groups):
        assert group.bootstrap_valid_repetitions == 20
        assert group.unavailable_reason is None
        assert group.effect == pytest.approx(point + shift)
        assert (group.lb, group.ub) == pytest.approx(tuple(bound + shift for bound in expected))
    # The old procedure erases ratio bias and reverses raw deviations.
    mean_star = sum(mean for mean, _ in samples) / len(samples)
    old = (
        point + mean_star - max(m for m, _ in samples),
        point + mean_star - min(m for m, _ in samples),
    )
    assert old != pytest.approx(expected)


def test_cluster_rank_response_reference_matches_integrated_weight_oracle():
    from increment.estimation.targeting import _cluster_mean, _rank_components

    # Two tied blocks each have probability 1/2. Their integrated AUTOC
    # weights are +/- log(2); Qini weights are +/- 1/4.
    score = np.tile([1.0, 0.0], 3)
    psi = np.array([4.0, 0.0, 8.0, 2.0, 6.0, -2.0])
    ids = np.repeat(np.arange(3), 2)
    autoc, qini, references = _rank_components(score, psi, np.ones(6))
    assert autoc == pytest.approx(3 * math.log(2))
    assert qini == pytest.approx(0.75)
    for reference, coefficient in zip(references, (math.log(2), 0.25), strict=True):
        expected = coefficient * np.tile([1.0, -1.0], 3) * (psi - 3.0)
        np.testing.assert_allclose(reference, expected)
        # Cluster sums are 4c, 6c, 8c, hence centered contributions -c/3, 0, c/3.
        summary = _cluster_mean(reference, np.ones(6), ids)
        assert summary.se == pytest.approx(coefficient / math.sqrt(3))
    shifted = _rank_components(score, psi + 2**40, np.ones(6))
    np.testing.assert_allclose(shifted[2], references)


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_delete_cluster_cv3_matches_ratio_leverage_identity(weighting):
    from fractions import Fraction

    from increment.estimation.targeting import (
        _cluster_deletions,
        _cluster_mean,
        _cluster_population,
        _delete_cluster_summary,
        _target_weights,
    )

    sizes, means = (1, 1, 1, 9), (0, 1, 2, 8)
    ids = np.repeat(np.arange(4), sizes)
    response = np.repeat(means, sizes).astype(float)
    population = _cluster_population(ids, np.zeros(ids.size), stratified=False)
    weights = _target_weights(ids, weighting)
    point = _cluster_mean(response, weights, ids)
    deleted = [
        _cluster_mean(response[rows], _target_weights(ids[rows], weighting), ids[rows]).value
        for rows in _cluster_deletions(population)
    ]
    result = _delete_cluster_summary(point.value, deleted, population)
    masses = sizes if weighting == "member_count" else (1, 1, 1, 1)
    total = sum(masses)
    mean = Fraction(sum(m * y for m, y in zip(masses, means, strict=True)), total)
    influence = [Fraction(m, total) * (y - mean) for m, y in zip(masses, means, strict=True)]
    changes = [u / (1 - Fraction(m, total)) for u, m in zip(influence, masses, strict=True)]
    variance = Fraction(3, 4) * sum(change**2 for change in changes)
    assert result.se == pytest.approx(math.sqrt(float(variance)))
    assert result.df == 3
    assert result.reason is None
    assert point.value is not None
    np.testing.assert_allclose(point.value - np.array(deleted), [float(v) for v in changes])
    if weighting == "equal":
        assert result.se == pytest.approx(point.se)
    else:
        assert result.se is not None
        assert point.se is not None
        assert result.se > point.se
        assert changes[-1] == 4 * influence[-1]  # h=3/4, determined by observed mass.


def test_delete_cluster_cv3_combines_signed_contrast_before_squaring():
    from increment.estimation.targeting import (
        _cluster_contrast,
        _cluster_deletions,
        _cluster_population,
        _delete_cluster_summary,
    )

    ids = np.arange(3)
    low = np.array([0.0, 10.0, -4.0])
    high = low + np.array([1.0, 2.0, 4.0])
    population = _cluster_population(ids, np.zeros(3), stratified=False)

    def contrast(rows):
        return _cluster_contrast(
            high[rows], low[rows], np.ones(rows.size), np.ones(rows.size), ids[rows], ids[rows]
        ).value

    point = contrast(ids)
    deleted = [contrast(rows) for rows in _cluster_deletions(population)]
    assert point == pytest.approx(7 / 3)
    assert deleted == pytest.approx([3.0, 2.5, 1.5])
    summary = _delete_cluster_summary(point, deleted, population)
    assert summary.se == pytest.approx(math.sqrt(7) / 3)


def test_complete_cluster_deletions_move_quantile_groups_and_qini():
    from increment.estimation.targeting import (
        _cluster_population,
        _cluster_validation_deletions,
        _delete_cluster_summary,
        _Holdout,
        _replicate_frame,
    )

    ids = np.repeat(np.arange(3), 2)
    score = np.arange(6, dtype=float)
    response = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 20.0])
    holdout = _Holdout(
        n_train=6,
        score=score,
        y=response,
        d=np.tile([0.0, 1.0], 3),
        unit_ids=np.arange(6).astype(str),
        cluster_ids=ids,
        cols={},
        covariates=(),
    )
    population = _cluster_population(ids, holdout.d, stratified=False)
    frame = _replicate_frame(holdout, response, (), population, "score")
    deleted = _cluster_validation_deletions(frame, 2, "score")
    # A six-member median group has mean 10. The four-member deletion groups
    # have means 15, 15, 0; keeping the original cutoff would give 10, 15, 0.
    assert deleted[3] == pytest.approx([15.0, 15.0, 0.0])
    assert _delete_cluster_summary(10.0, deleted[3], population).se == pytest.approx(10.0)
    # Four-member Qini weights are (-3,-1,1,3)/8, then average responses.
    assert deleted[1] == pytest.approx([35 / 16, 35 / 16, 0.0])


@pytest.mark.parametrize("n_groups", [2, 3])
@pytest.mark.parametrize("weighting", ["member_count", "equal"])
@pytest.mark.parametrize("arm_summary", ["score", "welch"])
def test_replicate_copies_match_materialized_whole_cluster_draws(arm_summary, weighting, n_groups):
    from increment.estimation import targeting

    def frozen(y, d, X, unit_ids, cluster_ids):
        return y, np.ones(y.size, dtype=bool)

    rng = np.random.default_rng(61)
    sizes = np.array([3, 1, 4, 2, 6, 3, 5, 2])
    ids = np.repeat([f"c{g}" for g in range(sizes.size)], sizes)
    # Rounded scores tie within and across clusters; arms alternate inside clusters.
    score = np.round(rng.normal(size=ids.size), 1)
    y = rng.gamma(2.0, size=ids.size) + score
    holdout = targeting._Holdout(
        n_train=ids.size,
        score=score,
        y=y,
        d=np.arange(ids.size) % 2.0,
        cols={"spend": rng.normal(size=ids.size)},
        covariates=(SPEND,),
        unit_ids=np.arange(ids.size).astype(str),
        cluster_ids=ids,
        psi_fn=targeting._ipw_psi_fn if arm_summary == "welch" else frozen,
        cluster_weight=weighting,
    )
    psi = targeting._ipw_psi(y, holdout.d) if arm_summary == "welch" else y
    plan = targeting._clan_plan(holdout.cols, holdout.covariates)
    population = targeting._cluster_population(ids, holdout.d, stratified=False)
    frame = targeting._replicate_frame(holdout, psi, plan, population, arm_summary)
    # Repeated sources, omitted sources, and a single deletion.
    for copies in ([2, 0, 1, 3, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0], [0, 4, 1, 0, 2, 0, 0, 1]):
        copies = np.array(copies)
        rows = np.concatenate(
            [population.members[g] for g in np.repeat(np.arange(copies.size), copies)]
        )
        instances = np.repeat(np.arange(copies.sum()), np.repeat(sizes, copies))
        resampled = targeting._resampled_holdout(holdout, rows, instances)
        expected = targeting._cluster_tables(
            resampled,
            targeting._resampled_psi(resampled, psi, rows, arm_summary),
            n_groups,
            arm_summary,
            tuple((name, values[rows]) for name, values in plan),
        )
        rows_out = (*expected.groups, *expected.clan)
        replicate = targeting._replicate_statistics(frame, copies, n_groups, arm_summary)
        assert replicate.failures == ()
        assert replicate.values == pytest.approx(
            (
                expected.autoc,
                expected.qini,
                *(g.effect for g in expected.groups),
                *(c.diff for c in expected.clan),
            ),
            rel=1e-12,
            abs=1e-12,
        )
        assert replicate.scales == pytest.approx(
            (*expected.rank_scales, *(row.se for row in rows_out)), rel=1e-12, abs=1e-12
        )
        assert replicate.reasons[2:] == tuple(
            targeting._replicate_reason(row.unavailable_reason) for row in rows_out
        )


def test_jackknife_interval_hull_and_test_use_both_components():
    from scipy.stats import t

    from increment.estimation.targeting import _BootstrapResult, _ClusterSummary, _jackknife_hull

    bootstrap = _BootstrapResult(0.75, 0.01, 1.0, 100.0, 199, None)
    summary = _ClusterSummary(2.0, 3.0, 3.0, 4, "delete-cluster-CV3")
    result = _jackknife_hull(bootstrap, summary, 0.2, family=2)
    assert result.lb == pytest.approx(2 - 3 * t.isf(0.05, 3))
    assert result.ub == 100.0
    assert result.p_value == pytest.approx(t.sf(2 / 3, 3))
    assert result.se == bootstrap.se and result.valid == 199
    reversed_p = _jackknife_hull(bootstrap._replace(p_value=0.9), summary, 0.2, family=2)
    assert reversed_p.p_value == 0.9


@pytest.mark.parametrize("missing", [None, math.inf, math.nan])
def test_delete_cluster_does_not_drop_an_unavailable_deletion(missing):
    from increment.estimation.targeting import (
        _BootstrapResult,
        _cluster_population,
        _delete_cluster_summary,
        _jackknife_hull,
    )

    population = _cluster_population(np.arange(3), np.zeros(3), stratified=False)
    summary = _delete_cluster_summary(2.0, [1.0, missing, 3.0], population)
    bootstrap = _BootstrapResult(0.75, 0.1, 0.0, 4.0, 199, None)
    result = _jackknife_hull(bootstrap, summary, 0.2)
    assert result.valid == 199
    assert result.reason == "estimation.targeting.jackknife_unavailable_replicate"
    assert result.se is result.p_value is result.lb is result.ub is None


def test_delete_cluster_cv3_centers_at_full_statistic_and_limits_stratum_df():
    from increment.estimation.targeting import _cluster_population, _delete_cluster_summary

    ids = np.arange(5)
    population = _cluster_population(ids, np.array([0, 0, 1, 1, 1]), stratified=True)
    deleted = [1.0, 2.0, 3.0, 4.0, 5.0]
    summary = _delete_cluster_summary(4.0, deleted, population)
    # CV3 exceeds mean-centered CV3J by (K-1)*(mean(delete)-T)^2 = 4.
    assert summary.se == pytest.approx(math.sqrt(12.0))
    assert summary.df == 1
    shifted = _delete_cluster_summary(2**40 + 4.0, [2**40 + v for v in deleted], population)
    reordered = _delete_cluster_summary(4.0, deleted[::-1], population)
    assert shifted.se == reordered.se == summary.se


def test_delete_cluster_cv3_preserves_representable_scale_near_overflow():
    from increment.estimation.targeting import _cluster_population, _delete_cluster_summary

    population = _cluster_population(np.arange(2), np.zeros(2), stratified=False)
    summary = _delete_cluster_summary(1e308, [-1e308, 1e308], population)
    assert summary.reason is None
    assert summary.se is not None
    assert summary.se / 1e308 == pytest.approx(math.sqrt(2))


def test_bootstrap_t_scales_opposite_extreme_roots_before_subtraction():
    from increment.estimation.targeting import _bootstrap_bound, _centered_bootstrap

    values = [-1e308, 9e307, 9.1e307, 9.2e307, 9.3e307, 9.4e307, 9.5e307, 9.6e307, 9.7e307]
    result = _centered_bootstrap(
        1e308,
        values,
        0.8,
        scale=1e308,
        scales=[1e308] * len(values),
    )
    assert result.reason is None
    assert result.se is not None and math.isfinite(result.se)
    assert result.lb is not None and result.ub is not None
    assert _bootstrap_bound(1e308, 1e308, 2.0) == -1e308


def test_nonfinite_cluster_net_benefit_is_nullable_for_resampled_replicates():
    from increment.estimation.targeting import _cluster_net_benefits

    (result,) = _cluster_net_benefits(
        np.zeros(4),
        np.full(4, math.inf),
        np.zeros(4),
        np.tile([0.0, 1.0], 2),
        np.arange(4),
        np.zeros(4, dtype=int),
        (1.0,),
        0.0,
        "equal",
        randomized_score=False,
    )
    assert result.value is None
    assert result.reason == "estimation.targeting.nonfinite_statistic"


def test_cluster_selection_refuses_nonfinite_candidates_before_argmax(monkeypatch):
    from increment.errors import CodedError
    from increment.estimation import targeting
    from increment.estimation.targeting import _ClusterSummary

    def unavailable(*args, **_kwargs):
        return tuple(
            _ClusterSummary(
                None,
                None,
                None,
                4,
                "equal-cluster",
                "estimation.targeting.nonfinite_statistic",
            )
            for _ in args[6]
        )

    monkeypatch.setattr(targeting, "_cluster_net_benefits", unavailable)
    with pytest.raises(CodedError) as raised:
        _call_array_entry_point(
            "select_targeting_rule_arrays",
            _c08_fixture(mixed=True),
            interact=[SPEND],
            cluster_weight="equal",
            bootstrap_repetitions=2,
            n_groups=2,
        )
    assert raised.value.code == "estimation.targeting.nonfinite_statistic"


def test_jackknife_hull_preserves_exact_family_tail_and_unresolved_bootstrap(monkeypatch):
    from fractions import Fraction

    from increment.estimation import targeting

    seen = []

    def critical(tail, df):
        seen.append(tail)
        return 2.0

    monkeypatch.setattr(targeting, "student_t_isf", critical)
    summary = targeting._ClusterSummary(1.0, 2.0, 3.0, 4, "delete-cluster-CV3")
    bootstrap = targeting._BootstrapResult(0.5, 0.1, 0.0, 2.0, 199, None)
    result = targeting._jackknife_hull(bootstrap, summary, 0.1, family=3)
    assert (result.lb, result.ub) == (-3.0, 5.0)
    assert Fraction(seen[0]) <= Fraction(0.1) / 6
    assert Fraction(math.nextafter(seen[0], math.inf)) > Fraction(0.1) / 6
    unresolved = bootstrap._replace(
        lb=None, ub=None, reason="estimation.targeting.bootstrap_tail_resolution"
    )
    result = targeting._jackknife_hull(unresolved, summary, 1e-300)
    assert result.reason == unresolved.reason
    assert result.lb is result.ub is None
    assert result.p_value is not None
    assert bootstrap.p_value is not None
    assert result.p_value > bootstrap.p_value


def test_cluster_bootstrap_draw_multiplicity_is_separate_instances_and_preserves_arms():
    from increment.estimation.targeting import _cluster_draw, _cluster_population, _target_weights

    ids = np.repeat(["a", "b", "c", "d"], [1, 3, 2, 4])
    d = np.repeat([0.0, 0.0, 1.0, 1.0], [1, 3, 2, 4])
    # Locate a repeated draw; the assertion checks values/identities, not RNG internals.
    population = _cluster_population(ids, d, stratified=True)
    for seed in range(20):
        rows, instances = _cluster_draw(population, np.random.default_rng(seed))
        if np.unique(ids[rows]).size < 4:
            break
    else:
        pytest.fail("fixture should include a repeated source cluster")
    assert np.unique(instances).size == 4
    weights = _target_weights(instances, "equal")
    np.testing.assert_allclose(np.bincount(instances, weights=weights), np.ones(4))
    for instance in range(4):
        inside = instances == instance
        assert np.unique(ids[rows][inside]).size == 1
        assert inside.sum() == np.count_nonzero(ids == ids[rows][inside][0])
    assert sorted(d[rows][instances == g][0] for g in range(4)) == [0, 0, 1, 1]


def _c07_holdout(weighting="member_count"):
    from increment.estimation.cate import CateScoreState, DesignSpec
    from increment.estimation.targeting import _Holdout

    rng = np.random.default_rng(927)
    ids = np.repeat([f"g{i:02}" for i in range(12)], 4)
    score = np.tile([0.0, 1.0, 2.0, 3.0], 12)
    d = np.tile([0.0, 1.0, 0.0, 1.0], 12)
    y = 0.5 * score + d * (score + 0.2) + rng.normal(size=48)

    # A caller-supplied frozen score must retain the five-position interface.
    def frozen(y, d, X, unit_ids, cluster_ids):
        return y, np.ones(y.size, dtype=bool)

    holdout = _Holdout(
        n_train=48,
        score=score,
        y=y,
        d=d,
        cols={"spend": score},
        covariates=(SPEND,),
        unit_ids=np.array([f"u{i}" for i in range(48)]),
        cluster_ids=ids,
        psi_fn=frozen,
        cluster_weight=weighting,
        score_state=CateScoreState(
            basis=DesignSpec.model_validate(
                {
                    "transforms": [
                        {
                            "name": "spend",
                            "kind": "continuous",
                            "mean": 0.0,
                            "scale": 1.0,
                        }
                    ]
                }
            ),
            means=(0.0,),
            coefficients=(1.0,),
            intercept=0.0,
        ),
    )
    return holdout


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_cluster_validation_bootstrap_row_cloning_and_nullable_json(weighting):
    from increment.estimation.targeting import CateValidation, _validation

    holdout = _c07_holdout(weighting)
    original = _validation(
        holdout,
        holdout.y,
        n_groups=2,
        alpha=0.2,
        arm_summary="score",
        bootstrap_seed=57,
        bootstrap_repetitions=19,
    )
    assert (
        _validation(
            holdout,
            holdout.y,
            n_groups=2,
            alpha=0.2,
            arm_summary="score",
            bootstrap_seed=57,
            bootstrap_repetitions=19,
        )
        == original
    )
    cloned = holdout._replace(
        score=np.tile(holdout.score, 3),
        y=np.tile(holdout.y, 3),
        d=np.tile(holdout.d, 3),
        cols={name: np.tile(values, 3) for name, values in holdout.cols.items()},
        unit_ids=np.tile(holdout.unit_ids, 3),
        cluster_ids=np.tile(holdout.cluster_ids, 3),
    )
    repeated = _validation(
        cloned,
        cloned.y,
        n_groups=2,
        alpha=0.2,
        arm_summary="score",
        bootstrap_seed=57,
        bootstrap_repetitions=19,
    )
    assert original.holdout_ate is not None and repeated.holdout_ate is not None
    assert original.holdout_ate.value == pytest.approx(repeated.holdout_ate.value)
    assert original.holdout_ate_se == pytest.approx(repeated.holdout_ate_se)
    assert original.autoc.estimate == pytest.approx(repeated.autoc.estimate)
    assert original.autoc.se is not None
    assert original.autoc.se == pytest.approx(repeated.autoc.se)
    assert original.autoc.p_value == repeated.autoc.p_value
    for a, b in zip(original.groups, repeated.groups, strict=True):
        assert a.effect == pytest.approx(b.effect)
        assert a.se == pytest.approx(b.se)
    assert original.bootstrap_seed == 57 and original.bootstrap_repetitions == 19
    assert original.n_clusters == 12
    assert original.autoc.uncertainty_method == "bootstrap-t+cluster-jackknife-t"
    assert CateValidation.model_validate_json(original.model_dump_json()) == original
    assert "NaN" not in original.model_dump_json()


def test_cluster_validation_all_ties_and_empty_groups_have_precise_reasons():
    from increment.estimation.targeting import _validation

    holdout = _c07_holdout()._replace(score=np.zeros(48))
    result = _validation(
        holdout,
        holdout.y,
        n_groups=2,
        alpha=0.2,
        arm_summary="score",
        bootstrap_repetitions=9,
    )
    assert result.autoc.estimate == 0
    assert result.autoc.se is None and result.autoc.p_value is None
    assert result.autoc.unavailable_reason == "estimation.targeting.degenerate_rank_distribution"
    assert "unavailable_reason='estimation.targeting.degenerate_rank_distribution'" in repr(result)
    assert result.groups[1].effect is None and result.groups[1].se is None
    assert result.groups[1].unavailable_reason == "estimation.targeting.empty_group"
    assert result.passed is False
    assert "NaN" not in result.model_dump_json()


def test_cluster_validation_single_original_cluster_bootstrap_is_unavailable():
    from increment.estimation.targeting import _validation

    holdout = _c07_holdout()._replace(cluster_ids=np.repeat("only", 48))
    result = _validation(
        holdout,
        holdout.y,
        n_groups=2,
        alpha=0.2,
        arm_summary="score",
        bootstrap_repetitions=9,
    )
    assert result.holdout_ate is not None
    assert result.holdout_ate.value is not None
    assert result.holdout_ate.lb is None and result.holdout_ate_se is None
    assert result.autoc.p_value is None and result.autoc.se is None
    assert result.autoc.unavailable_reason == "estimation.targeting.insufficient_clusters"
    assert result.autoc.bootstrap_valid_repetitions == 0
    assert "NaN" not in result.model_dump_json()


def test_cluster_dr_frozen_nuisances_prevent_heldout_outcome_leakage():
    from increment.estimation.targeting import _freeze_psi_fn

    rng = np.random.default_rng(362)
    train_x = rng.normal(size=(120, 2))
    train_d = np.tile([0.0, 1.0], 60)
    train_y = train_x[:, 0] + train_d * (1 + train_x[:, 1]) + rng.normal(size=120)
    hold_x = rng.normal(size=(24, 2))
    hold_d = np.tile([0.0, 1.0], 12)
    hold_y = hold_x[:, 0] + hold_d + rng.normal(size=24)
    unit_ids = np.array([f"h{i}" for i in range(24)])
    cluster_ids = np.repeat([f"c{i}" for i in range(6)], 4)
    plan = functools.partial(
        _dr_psi,
        propensity_learner=LogisticPropensity,
        outcome_learner=RidgeOutcome,
        folds=3,
        seed=19,
        gate=IdentificationGate(min_propensity=0.01),
    )
    frozen = _freeze_psi_fn(plan, train_y, train_d, train_x, hold_x)
    before, kept = frozen(hold_y, hold_d, hold_x, unit_ids, cluster_ids)
    perturbed = hold_y.copy()
    perturbed[7] += 1000
    after, kept_after = frozen(perturbed, hold_d, hold_x, unit_ids, cluster_ids)
    np.testing.assert_array_equal(kept, kept_after)
    assert kept.all()
    # A held-out outcome moves its own score only: the nuisances stay frozen.
    assert before[7] != after[7]
    others = np.arange(24) != 7
    np.testing.assert_array_equal(before[others], after[others])


@pytest.mark.parametrize(
    "caller", ["validate_cate_arrays", "targeting_rule_arrays", "select_targeting_rule_arrays"]
)
def test_cluster_weight_and_bootstrap_options_refuse_before_any_fit(caller, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("invalid options must refuse before fitting")

    monkeypatch.setattr("increment.estimation.targeting.fit_cate", fail)
    with pytest.raises(InvalidRequestError) as error:
        _call_array_entry_point(caller, _signal_fixture(), interact=[SPEND], cluster_weight="equal")
    assert error.value.code == "estimation.cate.equal_weighting_without_cluster"
    with pytest.raises(InvalidRequestError) as error:
        _call_array_entry_point(
            caller, _signal_fixture(), interact=[SPEND], bootstrap_repetitions=1
        )
    assert error.value.code == "estimation.targeting.bootstrap_options"


def test_cluster_bootstrap_collapse_of_distinct_original_clusters_is_not_p_one(monkeypatch):
    from increment.estimation.targeting import _validation

    holdout = _c07_holdout()

    def collapsed(population, rng):
        # Twelve independent sampled instances, but only one source cluster.
        return np.zeros(12, dtype=int)

    monkeypatch.setattr("increment.estimation.targeting._cluster_resample", collapsed)
    result = _validation(
        holdout,
        holdout.y,
        n_groups=2,
        alpha=0.2,
        arm_summary="score",
        bootstrap_repetitions=9,
    )
    assert result.autoc.se is None and result.autoc.p_value is None
    assert result.autoc.lb is None and result.autoc.ub is None
    assert result.autoc.unavailable_reason == "estimation.targeting.bootstrap_zero_variance"
    assert result.autoc.bootstrap_valid_repetitions == 9


def test_cluster_weighted_rank_integrals_match_two_block_closed_form():
    from increment.estimation.targeting import _weighted_ranks

    # Half the target mass has effect four and the other half effect zero.
    # TOC(q)=2 for q<=1/2, then 2/q-2, giving AUTOC=2 log(2), Qini=1/2.
    score = np.array([1.0, 0.0, 0.0, 0.0])
    psi = np.array([4.0, 0.0, 0.0, 0.0])
    autoc, qini = _weighted_ranks(score, psi, np.array([3.0, 1.0, 1.0, 1.0]))
    assert autoc == pytest.approx(2 * math.log(2))
    assert qini == pytest.approx(0.5)


def test_cluster_policy_recomputes_empirical_cutoff_and_equal_weights_on_resample():
    from increment.estimation.targeting import _cluster_policy, _resampled_holdout

    holdout = _c07_holdout("equal")._replace(score=np.repeat(np.arange(12.0), 4))
    deployment, value, _ = _cluster_policy(holdout, holdout.y, 0.5, "score")
    # Resample the highest cluster eight times and four other clusters once.
    labels = [0, 1, 2, 3] + [11] * 8
    rows = np.concatenate([np.arange(g * 4, (g + 1) * 4) for g in labels])
    instances = np.repeat(np.arange(12), 4)
    resampled = _resampled_holdout(holdout, rows, instances)
    new_deployment, new_value, _ = _cluster_policy(resampled, holdout.y[rows], 0.5, "score")
    assert deployment.threshold == 5.0
    assert new_deployment.threshold == 11.0
    assert new_value.value == pytest.approx(holdout.y[-4:].mean())
    assert new_value.value != pytest.approx(value.value)


def test_cluster_honest_holdout_defaults_to_training_frozen_dr_scores():
    from increment.estimation.targeting import _holdout_scores

    rng = np.random.default_rng(712)
    ids = np.repeat([f"cluster-{i}" for i in range(32)], 4)
    units = np.array([f"unit-{i}" for i in range(128)])
    x = rng.normal(size=128)
    d = np.tile([0.0, 1.0], 64)
    y = 0.4 * x + d * (1 + x) + rng.normal(size=128)
    plan = functools.partial(
        _dr_psi,
        propensity_learner=LogisticPropensity,
        outcome_learner=RidgeOutcome,
        folds=3,
        seed=19,
        gate=IdentificationGate(min_propensity=0.01),
    )
    holdout = _honest_holdout(
        y,
        d,
        {"spend": x},
        units,
        cluster_ids=ids,
        interact=[SPEND],
        adjust=(),
        adjustment=("spend",),
        psi_fn=plan,
        n_groups=2,
        alpha=0.05,
    )
    before, kept = _holdout_scores(holdout)
    perturbed = holdout.y.copy()
    perturbed[0] += 500
    after, kept_after = _holdout_scores(holdout._replace(y=perturbed))
    assert kept.all() and kept_after.all()
    assert before[0] != after[0]
    np.testing.assert_array_equal(before[1:], after[1:])


@pytest.mark.parametrize(
    "caller", ["validate_cate_arrays", "targeting_rule_arrays", "select_targeting_rule_arrays"]
)
@pytest.mark.parametrize("repetitions", [9, 99])
def test_cluster_public_array_paths_thread_weighting_and_bootstrap_provenance(caller, repetitions):
    rng = np.random.default_rng(283)
    sizes = np.tile([4, 6, 8, 10], 8)
    ids = np.repeat([f"c{i}" for i in range(32)], sizes)
    unit_ids = np.array([f"u{i}" for i in range(ids.size)])
    d = np.concatenate([np.tile([0.0, 1.0], size // 2) for size in sizes])
    x = rng.normal(size=ids.size)
    y = x + d * (1 + x) + rng.normal(size=ids.size)
    result = _call_array_entry_point(
        caller,
        {"y": y, "d": d, "cols": {"spend": x}, "unit_ids": unit_ids, "cluster_ids": ids},
        interact=[SPEND],
        cluster_weight="equal",
        bootstrap_seed=23,
        bootstrap_repetitions=repetitions,
        n_groups=2,
    )
    if isinstance(result, TargetingSelection):
        assert result.cluster_weight == "equal"
        assert result.bootstrap_seed == 23 and result.bootstrap_repetitions == repetitions
        assert all(row.cluster_weight == "equal" for row in result.inner)
        selected = result.inner[result.fractions.index(result.selected_fraction)]
        assert result.bootstrap_valid_repetitions == selected.bootstrap_valid_repetitions
        assert result.unavailable_reason == selected.unavailable_reason
        assert selected.reference_df is not None
        assert 0 < selected.reference_df <= selected.n_clusters - 1
        assert result.reference_df == selected.reference_df
        assert result.support_failures == selected.support_failures
        assert result.bootstrap_valid_repetitions == repetitions
        assert (result.unavailable_reason is None) == (repetitions == 99)
        result = result.rule
    validation = result.validation if hasattr(result, "validation") else result
    assert validation.cluster_weight == "equal"
    assert validation.bootstrap_seed == 23 and validation.bootstrap_repetitions == repetitions
    assert validation.autoc.bootstrap_seed == 23
    assert validation.autoc.bootstrap_repetitions == repetitions
    assert all(group.bootstrap_repetitions == repetitions for group in validation.groups)
    assert "NaN" not in result.model_dump_json()


def test_cluster_bootstrap_unresolved_small_tail_is_nullable():
    from increment.estimation.targeting import _centered_bootstrap

    result = _centered_bootstrap(1.0, list(np.arange(20.0)), 1e-300, scale=1.0, scales=[1.0] * 20)
    assert result.se is not None and result.p_value is not None
    assert result.lb is None and result.ub is None
    assert result.reason == "estimation.targeting.bootstrap_tail_resolution"


@pytest.mark.parametrize("arm_summary", ["score", "welch"])
def test_cluster_public_validation_large_offset_identical_cluster_means(arm_summary):
    labels = np.array([f"offset-{i}" for i in range(32)])
    held = _holdout_mask(labels)
    train_ids = np.repeat(labels[~held][:8], 4)
    hold_ids = np.repeat(labels[held][:2], 2)
    ids = np.concatenate((train_ids, hold_ids))
    rng = np.random.default_rng(128)
    x = rng.normal(size=ids.size)
    d = np.tile([0.0, 1.0], ids.size // 2)
    y = x + d * (1 + x) + rng.normal(size=ids.size)
    a = float(2**53)
    y[-4:] = [a, a + 2, a, a + 2]

    def frozen(y, d, X, unit_ids, cluster_ids):
        return y, np.ones(y.size, dtype=bool)

    result = validate_cate_arrays(
        y,
        d,
        {"spend": x},
        np.array([f"u{i}" for i in range(ids.size)]),
        cluster_ids=ids,
        interact=[SPEND],
        psi_fn=frozen,
        arm_summary=arm_summary,
        n_groups=2,
        bootstrap_repetitions=2,
    )
    assert result.n_holdout == 4 and result.n_clusters == 2
    assert result.holdout_ate is not None
    assert result.holdout_ate.value == (a if arm_summary == "score" else 2.0)
    assert result.holdout_ate_se is None
    assert result.holdout_ate.lb is None and result.holdout_ate.ub is None
    assert result.unavailable_reason == "estimation.targeting.degenerate_cluster_variance"


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("shift", [0.0, float(2**53)])
def test_cluster_contrast_centers_before_restoring_offset(overlap, shift):
    from increment.estimation.targeting import _cluster_contrast

    a = shift + np.array([0.0, 2.0, 0.0, 2.0])
    b = shift + np.zeros(4)
    ga = np.array(["a", "a", "b", "b"])
    gb = ga if overlap else np.array(["c", "c", "d", "d"])
    result = _cluster_contrast(a, b, np.ones(4), np.ones(4), ga, gb)
    assert result.value == 1.0
    assert result.se is None
    assert result.reason == "estimation.targeting.degenerate_cluster_variance"
    # Moving the second cluster by four yields u=(-1,1), V=4 in either case.
    a[2:] += 4
    varied = _cluster_contrast(a, b, np.ones(4), np.ones(4), ga, gb)
    assert varied.value == 3.0
    assert varied.se == 2.0


@pytest.mark.slow
@pytest.mark.parametrize("deploy_grain", ["unit", "cluster"])
def test_cluster_selection_dr_scores_use_the_selection_training_complement(deploy_grain):
    from increment.estimation import targeting
    from increment.estimation.crossfit import fold_assignments, outer_split

    class HalfPropensity:
        def fit(self, X: np.ndarray, d: np.ndarray) -> None:
            pass

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(X.shape[0], 0.5)

    class MeanOutcome:
        def fit(self, X: np.ndarray, d: np.ndarray) -> None:
            self.mean = float(d.mean())

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(X.shape[0], self.mean)

    rng = np.random.default_rng(712)
    ids = np.repeat([f"cluster-{i}" for i in range(32)], 4)
    units = np.array([f"unit-{i}" for i in range(ids.size)])
    x = rng.normal(size=ids.size)
    d = np.tile([0.0, 1.0], ids.size // 2)
    y = 0.4 * x + d * (1 + x) + rng.normal(size=ids.size)
    inner = ~outer_split(units, test_size=0.5, seed=11, stratify=d, cluster_ids=ids)
    folds = fold_assignments(
        units[inner], n_folds=2, seed=11, stratify=d[inner], cluster_ids=ids[inner]
    )
    nuisance_folds = fold_assignments(
        units[inner],
        n_folds=3,
        seed=19,
        stratify=d[inner],
        cluster_ids=ids[inner],
    )
    treated = np.flatnonzero(d[inner] == 1)
    i, j = next(
        (int(i), int(j))
        for i in treated
        for j in treated
        if folds[i] == folds[j] and nuisance_folds[i] != nuisance_folds[j]
    )
    k = int(next(k for k in treated if folds[k] != folds[i]))
    inner_rows = np.flatnonzero(inner)
    heldout_change, training_change = y.copy(), y.copy()
    heldout_change[inner_rows[j]] += 40
    training_change[inner_rows[k]] += 40
    plan = functools.partial(
        _dr_psi,
        propensity_learner=HalfPropensity,
        outcome_learner=MeanOutcome,
        folds=3,
        seed=19,
        gate=IdentificationGate(min_propensity=0.01),
    )
    # The former full-inner cross-fit lets j change i despite sharing a selection fold.
    legacy, _ = plan(y[inner], d[inner], x[inner, None], units[inner], ids[inner])
    contaminated, _ = plan(
        heldout_change[inner], d[inner], x[inner, None], units[inner], ids[inner]
    )
    assert contaminated[i] != legacy[i]

    for outcome in (y, heldout_change, training_change):
        result = select_targeting_rule_arrays(
            outcome,
            d,
            {"spend": x},
            units,
            cluster_ids=ids,
            deploy_grain=deploy_grain,
            interact=[SPEND],
            adjustment=("spend",),
            psi_fn=plan,
            arm_summary="score",
            fractions=(0.5,),
            n_folds=2,
            seed=11,
            n_groups=2,
            bootstrap_repetitions=2,
        )
        assert result.population is None
    # The selection score is the surface under test; read it from the step that
    # builds it rather than spying on the table that consumes it.
    before, same_fold, training_fold = (
        targeting._selection_nuisance_scores(
            plan,
            outcome[inner],
            d[inner],
            x[inner, None],
            units[inner],
            ids[inner],
            n_folds=2,
            seed=11,
            layout=CovariateLayout(("spend",), (None,)),
        )[0]
        for outcome in (y, heldout_change, training_change)
    )
    unchanged = (folds == folds[i]) & (np.arange(folds.size) != j)
    np.testing.assert_array_equal(before[unchanged], same_fold[unchanged])
    assert same_fold[j] - before[j] == pytest.approx(80.0)
    n_training_treated = np.count_nonzero((folds != folds[i]) & (d[inner] == 1))
    assert training_fold[i] - before[i] == pytest.approx(-40 / n_training_treated)


def test_unclustered_selection_dr_scores_use_the_selection_training_complement():
    from increment.estimation import targeting
    from increment.estimation.crossfit import fold_assignments, outer_split

    class HalfPropensity:
        def fit(self, X: np.ndarray, d: np.ndarray) -> None:
            pass

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(X.shape[0], 0.5)

    class MeanOutcome:
        def fit(self, X: np.ndarray, d: np.ndarray) -> None:
            self.mean = float(d.mean())

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(X.shape[0], self.mean)

    rng = np.random.default_rng(712)
    units = np.array([f"unit-{i}" for i in range(128)])
    x = rng.normal(size=units.size)
    d = np.tile([0.0, 1.0], units.size // 2)
    y = 0.4 * x + d * (1 + x) + rng.normal(size=units.size)
    inner = ~outer_split(units, test_size=0.5, seed=11, stratify=d, cluster_ids=None)
    folds = fold_assignments(units[inner], n_folds=2, seed=11, stratify=d[inner], cluster_ids=None)
    nuisance_folds = fold_assignments(
        units[inner],
        n_folds=3,
        seed=19,
        stratify=d[inner],
        cluster_ids=None,
    )
    treated = np.flatnonzero(d[inner] == 1)
    i, j = next(
        (int(i), int(j))
        for i in treated
        for j in treated
        if folds[i] == folds[j] and nuisance_folds[i] != nuisance_folds[j]
    )
    k = int(next(k for k in treated if folds[k] != folds[i]))
    inner_rows = np.flatnonzero(inner)
    heldout_change, training_change = y.copy(), y.copy()
    heldout_change[inner_rows[j]] += 40
    training_change[inner_rows[k]] += 40
    plan = functools.partial(
        _dr_psi,
        propensity_learner=HalfPropensity,
        outcome_learner=MeanOutcome,
        folds=3,
        seed=19,
        gate=IdentificationGate(min_propensity=0.01),
    )
    # Full-inner nuisance cross-fitting leaks between units in the same selection fold.
    legacy, _ = plan(y[inner], d[inner], x[inner, None], units[inner], None)
    contaminated, _ = plan(heldout_change[inner], d[inner], x[inner, None], units[inner], None)
    assert contaminated[i] != legacy[i]

    for outcome in (y, heldout_change, training_change):
        result = select_targeting_rule_arrays(
            outcome,
            d,
            {"spend": x},
            units,
            cluster_ids=None,
            interact=[SPEND],
            adjustment=("spend",),
            psi_fn=plan,
            arm_summary="score",
            fractions=(0.5,),
            n_folds=2,
            seed=11,
            n_groups=2,
        )
        assert result.population is None
    # The selection score is the surface under test; read it from the step that
    # builds it rather than spying on the freezer that feeds it.
    before, same_fold, training_fold = (
        targeting._selection_nuisance_scores(
            plan,
            outcome[inner],
            d[inner],
            x[inner, None],
            units[inner],
            None,
            n_folds=2,
            seed=11,
            layout=CovariateLayout(("spend",), (None,)),
        )[0]
        for outcome in (y, heldout_change, training_change)
    )
    unchanged = (folds == folds[i]) & (np.arange(folds.size) != j)
    np.testing.assert_array_equal(before[unchanged], same_fold[unchanged])
    assert same_fold[j] - before[j] == pytest.approx(80.0)
    n_training_treated = np.count_nonzero((folds != folds[i]) & (d[inner] == 1))
    assert training_fold[i] - before[i] == pytest.approx(-40 / n_training_treated)


def test_cluster_inner_bootstrap_reports_fold_arm_support_before_ipw():
    from increment.estimation import targeting

    # Fold 0 has mixed M (two rows per arm) and control-only C.
    ids = np.array(["M"] * 4 + ["C"] * 2 + ["A"] * 4 + ["B"] * 4)
    d = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0] + [0.0, 1.0] * 4)
    folds = np.array([0] * 6 + [1] * 8)
    y = np.arange(14.0) ** 2
    score = np.tile([0.0, 1.0], 7)
    (result,) = targeting._cluster_inner_table(
        score,
        np.zeros(14),
        y,
        d,
        ids,
        folds,
        (0.5,),
        0.0,
        "member_count",
        randomized_score=True,
        best_index=0,
        alpha=0.2,
        seed=0,
        repetitions=3,
        arm_summary="welch",
    )
    assert math.isfinite(result.net_benefit.value)
    assert result.net_benefit.lb is None and result.net_benefit.ub is None
    assert result.unavailable_reason == "estimation.targeting.insufficient_arm_clusters"
    assert result.bootstrap_valid_repetitions == 0
    counts = {
        (f.stage, f.fold, f.arm, f.reason.rsplit(".", 1)[-1]): f.count
        for f in result.support_failures
    }
    assert counts == {
        ("original", 0, 1, "insufficient_arm_clusters"): 1,
    }
    assert all(f.fold == 0 for f in result.support_failures)


def test_cluster_inner_bootstrap_accepts_multiplicity_with_sufficient_sources(monkeypatch):
    from increment.estimation import targeting

    ids = np.repeat(["a", "b", "c", "d", "e", "f"], 4)
    d = np.tile([0.0, 1.0, 0.0, 1.0], 6)
    folds = np.repeat([0, 1], 12)
    score = np.tile([0.0, 1.0, 0.0, 1.0], 6)
    y = d * np.repeat([1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 4)
    draws = iter(((0, 0, 1, 3, 3, 4), (1, 2, 2, 4, 5, 5)))

    def draw(population, rng):
        rows = np.concatenate([np.arange(4 * g, 4 * g + 4) for g in next(draws)])
        return rows, np.repeat(np.arange(6), 4)

    monkeypatch.setattr(targeting, "_cluster_draw", draw)
    (result,) = targeting._cluster_inner_table(
        score,
        np.zeros(24),
        y,
        d,
        ids,
        folds,
        (0.25,),
        0.0,
        "member_count",
        randomized_score=True,
        best_index=0,
        alpha=0.8,
        seed=0,
        repetitions=2,
        arm_summary="welch",
    )
    # Draw net benefits are 13/12 and 29/12, giving nonzero bootstrap spread.
    assert result.bootstrap_valid_repetitions == 2
    assert result.support_failures == () and result.unavailable_reason is None
    assert result.net_benefit.lb is not None and result.net_benefit.ub is not None
    assert result.net_benefit.lb < result.net_benefit.ub


def test_cluster_inner_bootstrap_keeps_point_when_a_replicate_overflows(monkeypatch):
    from increment.estimation import targeting

    ids = np.arange(12)
    d = np.array([0.0] * 8 + [1.0] * 4)
    y = d * 3.6e307
    bad_draw = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 8, 9])
    draws = iter((ids, bad_draw, ids))
    monkeypatch.setattr(targeting, "_cluster_draw", lambda population, rng: (next(draws), ids))

    def table(assignment, outcome):
        return targeting._cluster_inner_table(
            np.arange(12.0),
            np.zeros(12),
            outcome,
            assignment,
            ids,
            np.zeros(12, dtype=int),
            (1.0,),
            0.0,
            "member_count",
            randomized_score=True,
            best_index=0,
            alpha=0.8,
            seed=0,
            repetitions=3,
            arm_summary="welch",
        )[0]

    result = table(d, y)
    assert result.net_benefit.value == pytest.approx(3.6e307)
    assert result.net_benefit.lb is result.net_benefit.ub is None
    assert result.unavailable_reason == "estimation.targeting.bootstrap_unavailable_replicate"
    assert result.bootstrap_valid_repetitions == 2
    assert result.support_failures == ()
    with pytest.raises(InvalidRequestError) as exc:
        table(d[bad_draw], y[bad_draw])
    assert exc.value.code == "estimation.targeting.nonfinite_statistic"


def test_cluster_validation_and_policy_record_empty_bootstrap_arm(monkeypatch):
    from increment.estimation import targeting

    holdout = _c07_holdout()
    assert holdout.cluster_ids is not None
    d = holdout.d.copy()
    d[:8] = 0.0
    holdout = holdout._replace(d=d, psi_fn=targeting._ipw_psi_fn)

    def draw(population, rng):
        # Two control sources remain, so the former global K check would pass.
        return np.tile([0, 1], 6)

    monkeypatch.setattr(targeting, "_cluster_resample", draw)
    psi, _ = targeting._holdout_scores(holdout)
    validation = targeting._validation(
        holdout,
        psi,
        n_groups=2,
        alpha=0.2,
        arm_summary="welch",
        bootstrap_repetitions=2,
    )
    rule = targeting._cluster_rule(
        holdout,
        validation,
        psi,
        fraction=0.75,
        alpha=0.2,
        arm_summary="welch",
        intervention_grain="unit",
        required_columns=("spend",),
    )
    for result in (validation.autoc, rule):
        assert result.bootstrap_valid_repetitions == 0
        assert result.unavailable_reason == "estimation.targeting.bootstrap_empty_arm"
        assert len(result.support_failures) == 1
        failure = result.support_failures[0]
        assert failure.stage == "bootstrap" and failure.arm == 1 and failure.count == 2


def test_cluster_validation_accounts_for_finite_draws_with_unavailable_scales(monkeypatch):
    from increment.estimation import targeting

    holdout = _c07_holdout()
    y = holdout.y.copy()
    y[:8] = np.tile([0.0, 1.0, 0.0, 1.0], 2)
    holdout = holdout._replace(y=y)

    def draw(population, rng):
        return np.tile([0, 1], 6)

    monkeypatch.setattr(targeting, "_cluster_resample", draw)
    validation = targeting._validation(
        holdout,
        y,
        n_groups=2,
        alpha=0.2,
        arm_summary="score",
        bootstrap_repetitions=2,
    )
    rule = targeting._cluster_rule(
        holdout,
        validation,
        y,
        fraction=0.25,
        alpha=0.2,
        arm_summary="score",
        intervention_grain="unit",
        required_columns=("spend",),
    )
    for result in (*validation.groups, rule):
        assert result.bootstrap_valid_repetitions == 2
        assert result.support_failures == ()
        assert result.unavailable_reason == "estimation.targeting.bootstrap_zero_variance"


def test_cluster_rank_and_ipw_centering_preserve_large_offset_variation():
    from increment.estimation.targeting import _weighted_ranks

    y = np.array([0.0, 2.0, 0.0, 2.0])
    shifted = y + float(2**53)
    d = np.array([0.0, 1.0, 0.0, 1.0])
    weights = np.ones(4)
    np.testing.assert_array_equal(
        _ipw_psi(y, d, weights=weights), _ipw_psi(shifted, d, weights=weights)
    )
    assert _weighted_ranks(d, y, weights) == _weighted_ranks(d, shifted, weights)


def test_two_cluster_qini_bootstrap_retains_repeated_source_draws():
    from increment.estimation.targeting import _validation, _weighted_ranks

    full = _c07_holdout()
    holdout = full._replace(
        score=full.score[:8],
        y=full.y[:8],
        d=full.d[:8],
        cols={name: values[:8] for name, values in full.cols.items()},
        unit_ids=full.unit_ids[:8],
        cluster_ids=full.cluster_ids[:8],
    )
    repetitions, seed = 99, 777
    rng = np.random.default_rng(seed)
    oracle = []
    for _ in range(repetitions):
        draw = rng.choice(2, size=2, replace=True)
        rows = np.concatenate([np.arange(4 * g, 4 * g + 4) for g in draw])
        score, response = holdout.score[rows], holdout.y[rows]
        # Qini is the trapezoidal area of cumulative centered response.
        cumulative_mass = [0.0]
        cumulative_response = [0.0]
        for level in sorted(set(score), reverse=True):
            inside = score == level
            cumulative_mass.append(cumulative_mass[-1] + inside.mean())
            cumulative_response.append(
                cumulative_response[-1] + np.sum(response[inside] - response.mean()) / rows.size
            )
        oracle.append(
            sum(
                (right - left) * (before + after) / 2
                for left, right, before, after in zip(
                    cumulative_mass[:-1],
                    cumulative_mass[1:],
                    cumulative_response[:-1],
                    cumulative_response[1:],
                    strict=True,
                )
            )
        )
        assert _weighted_ranks(score, response, np.ones(rows.size))[1] == pytest.approx(oracle[-1])
    result = _validation(
        holdout,
        holdout.y,
        n_groups=2,
        alpha=0.2,
        arm_summary="score",
        bootstrap_seed=seed,
        bootstrap_repetitions=repetitions,
    )
    assert np.std(oracle, ddof=1) > 0
    assert result.qini.bootstrap_valid_repetitions == repetitions
    assert result.qini.se == pytest.approx(np.std(oracle, ddof=1))
    assert result.qini.lb is not None and result.qini.ub is not None
    assert result.qini.unavailable_reason is None


def _c08_fixture(*, mixed=False) -> dict:
    rng = np.random.default_rng(481)
    clusters = np.repeat([f"cluster-{g:03}" for g in range(64)], 6)
    x = np.repeat(np.linspace(-2, 2, 64), 6) + np.tile([-2, -1, 0, 0, 1, 2], 64)
    d = np.tile([0.0, 1.0], 192) if mixed else np.repeat(np.tile([0.0, 1.0], 32), 6)
    y = 0.2 * x + d * (2 + 3 * x) + rng.normal(0, 0.2, x.size)
    return {
        "y": y,
        "d": d,
        "cols": {"spend": x},
        "cluster_ids": clusters,
        "unit_ids": np.array([f"u-{i:04}" for i in range(x.size)]),
    }


def _c08_linear_state():
    from increment.estimation.cate import CateScoreState, DesignSpec

    return CateScoreState(
        basis=DesignSpec.model_validate(
            {
                "transforms": [
                    {
                        "name": "spend",
                        "kind": "continuous",
                        "mean": 0.0,
                        "scale": 1.0,
                    }
                ]
            }
        ),
        means=(0.0,),
        coefficients=(1.0,),
        intercept=0.0,
    )


def _c08_analytic_holdout(weighting="member_count"):
    from increment.estimation.targeting import _Holdout

    sizes = [2, 4, 2, 4]
    ids = np.repeat(["a", "b", "c", "d"], sizes)
    # Members straddle unit cutoffs even though the cluster means are 9,7,5,3.
    score = np.repeat([9.0, 7.0, 5.0, 3.0], sizes) + np.tile([-4.0, 4.0], 6)
    psi = np.repeat([8.0, 4.0, 2.0, -2.0], sizes)
    return _Holdout(
        n_train=12,
        score=score,
        y=psi,
        d=np.tile([0.0, 1.0], 6),
        cols={"spend": score},
        covariates=(SPEND,),
        unit_ids=np.array([f"u{i}" for i in range(12)]),
        cluster_ids=ids,
        cluster_weight=weighting,
        score_state=_c08_linear_state(),
    )


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_cluster_deployment_exact_prefix_ties_boundaries_and_oversized_head(weighting):
    from increment.estimation._deployment import cluster_prefix

    ids = np.array(["z", "z", "z", "a", "b", "c"])
    score = np.array([10.0, 8.0, 9.0, 8.0, 8.0, 1.0])
    assert not cluster_prefix(score, ids, 0, weighting).targeted.any()
    assert cluster_prefix(score, ids, 1, weighting).targeted.all()
    # No backfill from smaller clusters when the highest-scoring cluster is too large.
    head_share = 0.5 if weighting == "member_count" else 0.25
    below = cluster_prefix(score, ids, np.nextafter(head_share, 0), weighting)
    assert not below.targeted.any() and below.achieved_fraction == 0
    first = cluster_prefix(score, ids, head_share, weighting)
    np.testing.assert_array_equal(first.targeted, ids == "z")
    fraction = 4 / 6 if weighting == "member_count" else 0.5
    tied = cluster_prefix(score, ids, fraction, weighting)
    np.testing.assert_array_equal(tied.targeted, np.isin(ids, ["z", "a"]))
    assert tied.achieved_fraction == fraction
    permutation = np.array([5, 4, 3, 2, 1, 0])
    shuffled = cluster_prefix(score[permutation], ids[permutation], fraction, weighting)
    np.testing.assert_array_equal(shuffled.targeted, tied.targeted[permutation])
    cloned = cluster_prefix(np.tile(score, 3), np.tile(ids, 3), fraction, weighting)
    np.testing.assert_array_equal(cloned.targeted, np.tile(tied.targeted, 3))


@pytest.mark.parametrize(
    "weighting,expected,net",
    [
        ("member_count", 16 / 3, 7 / 6),
        ("equal", 6.0, 1.5),
    ],
)
def test_cluster_policy_and_selected_fraction_match_analytic_weighted_targets(
    weighting, expected, net
):
    from increment.estimation.targeting import (
        _cluster_inner_table,
        _cluster_policy,
        _rule_at_fraction,
    )

    holdout = _c08_analytic_holdout(weighting)
    assert holdout.cluster_ids is not None
    _, value, uplift = _cluster_policy(holdout, holdout.y, 0.5, "score", "cluster")
    assert value.value == pytest.approx(expected)
    assert uplift.value == pytest.approx(3.0)
    table = _cluster_inner_table(
        holdout.score,
        holdout.y,
        holdout.y,
        holdout.d,
        holdout.cluster_ids,
        np.zeros(12, dtype=int),
        (0.25, 0.5, 0.75, 1.0),
        3.0,
        weighting,
        randomized_score=False,
        deploy_grain="cluster",
        best_index=1,
        alpha=0.8,
        seed=19,
        repetitions=9,
        arm_summary="score",
    )
    assert np.argmax([row.net_benefit.value for row in table]) == 1
    assert table[1].net_benefit.value == pytest.approx(net)
    assert table[1].achieved_fraction == 0.5
    assert table[1].net_benefit.lb is None
    # Fix the already-tested evidence gate to isolate the selected policy's estimand.
    validation = _validation(
        holdout,
        holdout.y,
        n_groups=2,
        alpha=0.8,
        arm_summary="score",
        bootstrap_repetitions=9,
    ).model_copy(update={"passed": True})
    rule = _rule_at_fraction(
        holdout,
        validation,
        holdout.y,
        fraction=table[1].fraction,
        alpha=0.8,
        arm_summary="score",
        intervention_grain="cluster",
        deploy_grain="cluster",
    )
    assert rule.policy_value is not None
    assert rule.policy_value.value == pytest.approx(expected)
    np.testing.assert_array_equal(
        rule.predict(holdout.cols, cluster_ids=holdout.cluster_ids),
        np.isin(holdout.cluster_ids, ["a", "b"]),
    )


@pytest.mark.parametrize("fraction", [0.0, 0.4, 1.0])
def test_cluster_deployment_training_and_unseen_actions_are_uniform_and_portable(fraction):
    from increment.estimation.targeting import TargetingRule

    fixture = _c08_fixture()
    rule = targeting_rule_arrays(
        **fixture,
        interact=[SPEND],
        fraction=fraction,
        intervention_grain="cluster",
        n_groups=2,
        alpha=0.4,
        bootstrap_repetitions=9,
    )
    assert rule.deploy_grain == "cluster" and rule.budget_rule == "whole_cluster_prefix"
    assert rule.achieved_fraction <= fraction
    reloaded = TargetingRule.model_validate_json(rule.model_dump_json())
    unseen_ids = np.array(["new-b", "new-a", "new-b", "new-a", "new-c", "new-c"])
    unseen_cols = {"spend": np.array([-3.0, 2.0, 7.0, 0.0, -4.0, 1.0])}
    for cols, ids in ((fixture["cols"], fixture["cluster_ids"]), (unseen_cols, unseen_ids)):
        action = rule.predict(cols, cluster_ids=ids)
        np.testing.assert_array_equal(action, reloaded.predict(cols, cluster_ids=ids))
        for cluster in np.unique(ids):
            assert np.unique(action[ids == cluster]).size == 1
        if fraction in (0, 1):
            assert np.all(action == bool(fraction))
    assert reloaded.score_state == rule.score_state


# 1% is below the smallest cluster's share of the holdout, so the
# highest-scoring cluster alone is oversized and the prefix stays empty.
@pytest.mark.parametrize("fraction", [0.0, 0.01])
def test_empty_cluster_prefix_reports_no_conditional_effect_and_predicts_empty(fraction):
    from increment.estimation.targeting import TargetingRule

    fixture = _c08_fixture()
    rule = targeting_rule_arrays(
        **fixture,
        interact=[SPEND],
        fraction=fraction,
        intervention_grain="cluster",
        n_groups=2,
        alpha=0.8,
        arm_summary="score",
        bootstrap_repetitions=9,
    )
    holdout = _content_hash_holdout(fixture["unit_ids"], cluster_ids=fixture["cluster_ids"])
    cols = {name: values[holdout] for name, values in fixture["cols"].items()}
    ids = fixture["cluster_ids"][holdout]
    assert rule.validation.passed
    assert rule.achieved_fraction == 0
    assert rule.policy_value is None and rule.uplift_vs_average is None
    assert rule.unavailable_reason == "estimation.targeting.empty_group"
    assert not rule.predict(cols, cluster_ids=ids).any()
    restored = TargetingRule.model_validate_json(rule.model_dump_json())
    assert restored.recommendation == "target" and restored.threshold is None
    assert not restored.predict(cols, cluster_ids=ids).any()


def test_mixed_dependence_clusters_default_to_unit_actions_with_clustered_uncertainty():
    fixture = _c08_fixture(mixed=True)
    rule = targeting_rule_arrays(
        **fixture,
        interact=[SPEND],
        fraction=0.5,
        arm_summary="score",
        n_groups=2,
        alpha=0.4,
        bootstrap_repetitions=9,
    )
    assert rule.deploy_grain == rule.intervention_grain == "unit"
    assert rule.uncertainty_method == "bootstrap-t+cluster-jackknife-t"
    assert rule.reference_df is not None
    assert rule.n_clusters is not None
    assert 0 < rule.reference_df <= rule.n_clusters - 1
    assert rule.validation.holdout_ate_se is not None
    actions = rule.predict(fixture["cols"], cluster_ids=fixture["cluster_ids"])
    np.testing.assert_array_equal(actions, rule.predict(fixture["cols"]))
    assert any(
        np.unique(actions[fixture["cluster_ids"] == g]).size == 2
        for g in np.unique(fixture["cluster_ids"])
    )
    # Pure arm dependence alone must also retain the unit default.
    pure = targeting_rule_arrays(
        **_c08_fixture(),
        interact=[SPEND],
        fraction=0.5,
        n_groups=2,
        alpha=0.4,
        bootstrap_repetitions=9,
    )
    assert pure.deploy_grain == "unit"


@pytest.mark.parametrize("caller", [targeting_rule_arrays, select_targeting_rule_arrays])
def test_incompatible_unit_deployment_refuses_before_fit(caller, monkeypatch):
    from increment.estimation import targeting

    def no_fit(*args, **kwargs):
        pytest.fail("incompatible deployment reached fitting")

    monkeypatch.setattr(targeting, "fit_cate", no_fit)
    options: dict = (
        {"fraction": 0.5} if caller is targeting_rule_arrays else {"fractions": (0.5,), "seed": 7}
    )
    with pytest.raises(InvalidRequestError) as caught:
        caller(
            **_c08_fixture(),
            interact=[SPEND],
            intervention_grain="cluster",
            deploy_grain="unit",
            **options,
        )
    assert caught.value.code == "estimation.targeting.unsupported_unit_deployment"
    with pytest.raises(TypeError):
        cast("dict[str, object]", caught.value.context)["override"] = True


def test_cluster_prediction_validates_grain_and_ids_before_scoring(monkeypatch):
    from increment.estimation.cate import CateScoreState

    rule = targeting_rule_arrays(
        **_c08_fixture(),
        interact=[SPEND],
        fraction=0.5,
        intervention_grain="cluster",
        n_groups=2,
        alpha=0.4,
        bootstrap_repetitions=9,
    )

    def no_score(*args, **kwargs):
        pytest.fail("invalid deployment reached scoring")

    monkeypatch.setattr(CateScoreState, "score", no_score)
    with pytest.raises(InvalidRequestError) as caught:
        rule.predict({"spend": np.ones(2)})
    assert caught.value.code == "estimation.targeting.deployment_cluster_ids_required"
    with pytest.raises(InvalidRequestError) as caught:
        rule.predict({"spend": np.ones(2)}, cluster_ids=np.array(["a", "b"]), deploy_grain="unit")
    assert caught.value.code == "estimation.targeting.unsupported_unit_deployment"
    with pytest.raises(InvalidRequestError) as caught:
        rule.predict({"spend": np.ones(2)}, cluster_ids=np.array(["a"]))
    assert caught.value.code == "estimation.targeting.deployment_cluster_ids_shape"
    with pytest.raises(InvalidRequestError):
        rule.predict({"spend": np.ones(2)}, cluster_ids=np.array([1, "1"], dtype=object))


def test_cluster_bootstrap_recomputes_pooled_prefix_on_distinct_draw_instances(monkeypatch):
    from increment.estimation import targeting

    holdout = _c08_analytic_holdout("equal")
    assert holdout.cluster_ids is not None
    validation = _validation(
        holdout,
        holdout.y,
        n_groups=2,
        alpha=0.8,
        arm_summary="score",
        bootstrap_repetitions=2,
    ).model_copy(update={"passed": True})
    draws = iter(((0, 0, 1, 3), (1, 2, 2, 3)))
    members = [np.flatnonzero(holdout.cluster_ids == g) for g in ("a", "b", "c", "d")]

    def draw(population, rng):
        chosen = next(draws)
        return (
            np.concatenate([members[g] for g in chosen]),
            np.concatenate([np.full(members[g].size, i) for i, g in enumerate(chosen)]),
        )

    original = targeting._cluster_policy
    values: dict[tuple[str, ...], float | None] = {}

    def capture(sample, *args, **kwargs):
        deployment, value, uplift = original(sample, *args, **kwargs)
        sources = sample.deployment_source_ids
        if sources is None:
            sources = sample.cluster_ids
        values[tuple(sorted(map(str, sources.tolist())))] = value.value
        return deployment, value, uplift

    monkeypatch.setattr(targeting, "_cluster_draw", draw)
    monkeypatch.setattr(targeting, "_cluster_policy", capture)
    rule = targeting._cluster_rule(
        holdout,
        validation,
        holdout.y,
        fraction=0.5,
        alpha=0.8,
        arm_summary="score",
        intervention_grain="cluster",
        deploy_grain="cluster",
        required_columns=("spend",),
    )
    assert rule.policy_value is not None and rule.policy_value.value == pytest.approx(6.0)
    # Each draw re-pools the prefix on its own instances: the first selects two
    # A instances, the second selects B and one C.
    assert values[("a",) * 4 + ("b",) * 4 + ("d",) * 4] == pytest.approx(8.0)
    assert values[("b",) * 4 + ("c",) * 4 + ("d",) * 4] == pytest.approx(3.0)
    assert rule.bootstrap_valid_repetitions == 2
    assert rule.support_failures == ()


def test_unclustered_policy_prediction_preserves_frozen_threshold_and_json():
    from increment.estimation.targeting import TargetingRule

    fixture = _step_fixture()
    rule = targeting_rule_arrays(**fixture, interact=[SPEND], n_groups=4, fraction=0.4)
    assert rule.threshold is not None
    unseen = {"spend": np.linspace(-10, 20, 31)}
    expected = rule.score_state.score(unseen) >= rule.threshold
    np.testing.assert_array_equal(rule.predict(unseen), expected)
    restored = TargetingRule.model_validate_json(rule.model_dump_json())
    np.testing.assert_array_equal(restored.predict(unseen), expected)
    assert restored.deploy_grain == "unit"


@pytest.mark.parametrize(
    "weighting,draw_values",
    [
        ("member_count", [7 / 6, 5 / 3, 1 / 6]),
        ("equal", [1.5, 2.5, 0.0]),
    ],
)
def test_inner_cluster_bootstrap_rebuilds_prefix_budget(weighting, draw_values, monkeypatch):
    from increment.estimation import targeting

    holdout = _c08_analytic_holdout(weighting)
    assert holdout.cluster_ids is not None
    members = [np.flatnonzero(holdout.cluster_ids == g) for g in ("a", "b", "c", "d")]
    draws = iter(((0, 0, 1, 3), (1, 2, 2, 3)))

    def draw(population, rng):
        chosen = next(draws)
        return (
            np.concatenate([members[g] for g in chosen]),
            np.concatenate([np.full(members[g].size, i) for i, g in enumerate(chosen)]),
        )

    monkeypatch.setattr(targeting, "_cluster_draw", draw)
    table = targeting._cluster_inner_table(
        holdout.score,
        holdout.y,
        holdout.y,
        holdout.d,
        holdout.cluster_ids,
        np.zeros(12, dtype=int),
        (0.5,),
        3.0,
        weighting,
        randomized_score=False,
        deploy_grain="cluster",
        best_index=0,
        alpha=0.8,
        seed=0,
        repetitions=2,
        arm_summary="score",
    )
    row = table[0]
    # The two injected draws are the only source of resampled spread, so a
    # replicate that reused the point's prefix budget leaves it degenerate.
    assert row.net_benefit.value == pytest.approx(draw_values[0])
    assert row.bootstrap_valid_repetitions == 2
    assert row.net_benefit.lb is not None and row.net_benefit.ub is not None
    assert row.net_benefit.lb < row.net_benefit.value < row.net_benefit.ub


def test_cluster_selection_outer_outcomes_cannot_change_fit_budget_or_inner_selection(monkeypatch):
    from increment.estimation import targeting
    from increment.estimation.crossfit import fold_assignments, outer_split

    fixture = _c08_fixture()
    ids, units, d = fixture["cluster_ids"], fixture["unit_ids"], fixture["d"]
    outer = outer_split(units, test_size=0.5, seed=17, stratify=d, cluster_ids=ids)
    inner = ~outer
    folds = fold_assignments(
        units[inner], n_folds=2, seed=17, stratify=d[inner], cluster_ids=ids[inner]
    )
    inner_ids, outer_ids = set(ids[inner]), set(ids[outer])
    assert inner_ids.isdisjoint(outer_ids)
    for label in (0, 1):
        assert set(ids[inner][folds == label]).isdisjoint(set(ids[inner][folds != label]))
    seen = []
    fit = targeting.fit_cate

    def record_fit(*args, **kwargs):
        seen.append(set(kwargs["cluster_ids"]))
        return fit(*args, **kwargs)

    monkeypatch.setattr(targeting, "fit_cate", record_fit)
    options: dict = {
        "interact": [SPEND],
        "fractions": (0, 0.25, 0.5, 1),
        "cost_per_treated": 2.0,
        "n_folds": 2,
        "seed": 17,
        "n_groups": 2,
        "intervention_grain": "cluster",
        "alpha": 0.4,
        "bootstrap_repetitions": 9,
    }
    before = select_targeting_rule_arrays(**fixture, **options)
    y = fixture["y"].copy()
    y[outer] += np.linspace(-100, 100, int(outer.sum()))
    after = select_targeting_rule_arrays(**(fixture | {"y": y}), **options)
    assert before.selected_fraction == after.selected_fraction
    assert before.inner == after.inner
    assert before.rule.score_state == after.rule.score_state
    np.testing.assert_array_equal(
        before.rule.predict(fixture["cols"], cluster_ids=ids),
        after.rule.predict(fixture["cols"], cluster_ids=ids),
    )
    assert before.rule.validation.holdout_ate != after.rule.validation.holdout_ate
    expected = [set(ids[inner][folds != label]) for label in (0, 1)] + [inner_ids]
    assert seen == expected * 2
    assert all(training.isdisjoint(outer_ids) for training in seen)


def test_policy_nested_state_is_immutable_and_copies_caller_owned_metadata():
    from pydantic import ValidationError

    from increment.estimation.targeting import TargetingRule

    rule = targeting_rule_arrays(
        **_c08_fixture(),
        interact=[SPEND],
        fraction=0.5,
        intervention_grain="cluster",
        n_groups=2,
        alpha=0.4,
        bootstrap_repetitions=9,
    )
    payload = rule.model_dump(mode="json")
    rebuilt = TargetingRule.model_validate(payload)
    cols, ids = {"spend": np.array([4.0, 6.0, -2.0, 0.0])}, np.array(["a", "a", "b", "b"])
    expected = rebuilt.predict(cols, cluster_ids=ids)
    payload["score_state"]["coefficients"][0] *= -100
    payload["score_state"]["basis"]["transforms"][0]["mean"] += 100
    payload["required_columns"].append("injected")
    np.testing.assert_array_equal(rebuilt.predict(cols, cluster_ids=ids), expected)
    assert rebuilt.required_columns == ("spend",)
    with pytest.raises(ValidationError):
        cast("Any", rebuilt).deploy_grain = "unit"
    with pytest.raises(ValidationError):
        cast("Any", rebuilt.score_state.basis.transforms[0]).mean = 12
    with pytest.raises(TypeError):
        cast("list[float]", rebuilt.score_state.coefficients)[0] = 12
    with pytest.raises(InvalidRequestError) as caught:
        TargetingRule.model_validate(rule.model_dump() | {"budget_rule": "unit_threshold"})
    assert caught.value.code == "estimation.targeting.policy_metadata"


@pytest.mark.parametrize(
    "values",
    [
        [1e15, np.nextafter(1e15, np.inf), 1e15 + 0.25],
        [np.finfo(float).max, np.finfo(float).max, -np.finfo(float).max],
        [np.finfo(float).max] * 3,
    ],
)
def test_cluster_score_pooling_is_finite_and_order_independent_near_overflow(values):
    from decimal import Decimal, localcontext

    from increment.estimation._deployment import pool_cluster_scores

    score = np.array(values)
    ids = np.repeat("a", score.size)
    with localcontext() as context:
        context.prec = 400
        expected = float(sum(Decimal.from_float(float(v)) for v in values) / len(values))
    result = pool_cluster_scores(score, ids)[0].score
    assert math.isfinite(result)
    assert result == pytest.approx(expected, rel=2 * np.finfo(float).eps)
    assert pool_cluster_scores(score[::-1], ids)[0].score == result


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
@pytest.mark.parametrize("order_seed", [None, 72])
def test_cluster_roundoff_cancellation_uses_exact_binary64_totals(weighting, order_seed):
    from fractions import Fraction

    from increment.estimation.targeting import _cluster_contrast, _cluster_mean, _target_weights

    y = np.array([0.1, -0.1, 0.3, -0.3, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.1, -0.1])
    ids = np.repeat([0, 1, 2], [2, 4, 6])
    if order_seed is not None:
        order = np.random.default_rng(order_seed).permutation(y.size)
        y, ids = y[order], ids[order]
    weights = _target_weights(ids, weighting)
    assert all(sum(Fraction(float(v)) for v in y[ids == g]) == 0 for g in range(3))
    mean = _cluster_mean(y, weights, ids)
    assert mean.value == 0 and mean.se is None
    assert mean.reason == "estimation.targeting.degenerate_cluster_variance"
    for other_ids in (ids, ids + 3):
        contrast = _cluster_contrast(y, -y, weights, weights, ids, other_ids)
        assert contrast.value == 0 and contrast.se is None
        assert contrast.reason == mean.reason


@pytest.mark.parametrize("scale", [1.0, 1e-200, 1e300])
def test_cluster_roundoff_resolution_preserves_neighboring_float_effects(scale):
    from fractions import Fraction

    from increment.estimation.targeting import _cluster_contrast, _cluster_mean

    y = np.array([0.1, -0.1, 0.3, -0.3, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.1, -0.1]) * scale
    ids = np.repeat([0, 1, 2], [2, 4, 6])
    y[0] = np.nextafter(y[0], np.inf)
    exact = [Fraction(float(v)) for v in y]
    mu = sum(exact) / len(exact)
    u = [sum(v - mu for v, g in zip(exact, ids, strict=True) if g == j) / len(y) for j in range(3)]
    expected = math.sqrt(1.5) * math.hypot(*(float(v) for v in u))
    result = _cluster_mean(y, np.ones(12), ids)
    assert result.reason is None and result.se is not None
    assert result.se / expected == pytest.approx(1.0)
    contrast = _cluster_contrast(y, np.zeros(12), np.ones(12), np.ones(12), ids, ids)
    assert contrast.reason is None and contrast.se is not None
    assert contrast.se / expected == pytest.approx(1.0)


def test_cluster_bootstrap_roundoff_from_unequal_zero_sum_clusters():
    from increment.estimation.targeting import (
        _centered_bootstrap,
        _cluster_draw,
        _cluster_population,
        _weighted_mean,
        _weighted_ranks,
    )

    y = np.array([0.1, -0.1, 0.3, -0.3, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.1, -0.1])
    ids = np.repeat([0, 1, 2], [2, 4, 6])
    population = _cluster_population(ids, np.zeros(12), stratified=False)
    rng = np.random.default_rng(72)
    samples = [[], [], []]
    for _ in range(39):
        rows, _ = _cluster_draw(population, rng)
        weights = np.ones(rows.size)
        samples[0].append(_weighted_mean(y[rows], weights))
        autoc, qini = _weighted_ranks(ids[rows], y[rows], weights)
        samples[1].append(autoc)
        samples[2].append(qini)
    for values in samples:
        result = _centered_bootstrap(0.0, values, 0.05, scale=1.0, scales=[1.0] * 39)
        assert result.reason == "estimation.targeting.bootstrap_zero_variance"
        assert result.valid == 39
        assert result.se is result.p_value is result.lb is result.ub is None


@pytest.mark.parametrize("base", [0.1, 1e-200, 1e300])
def test_bootstrap_retains_neighboring_float_spread(base):
    from increment.estimation.targeting import _centered_bootstrap

    step = np.nextafter(base, np.inf) - base
    samples = [base, base + step] * 20
    result = _centered_bootstrap(base, samples, 0.1, scale=step, scales=[step] * 40)
    assert result.reason is None and result.se is not None
    assert result.se / step == pytest.approx(math.sqrt(40 / 39) / 2)


@pytest.mark.parametrize("one_extreme", [False, True])
def test_centered_bootstrap_preserves_representable_extreme_uncertainty(one_extreme):
    from increment.estimation.targeting import _centered_bootstrap

    # Roots are the draws themselves (T=0, unit scales); se is the draws' SD.
    if one_extreme:
        samples = [-1.7e308] + [1.7e308] * 39
        expected_se = 1.7e308 * (2 / math.sqrt(40))
        expected_bounds = (-1.7e308, -1.7e308)
        expected_p = 40 / 41
    else:
        samples = [-9e307, 9e307] * 20
        expected_se = 9e307 * math.sqrt(40 / 39)
        expected_bounds = (-9e307, 9e307)
        expected_p = 21 / 41
    result = _centered_bootstrap(0.0, samples, 0.1, scale=1.0, scales=[1.0] * 40)
    assert result.reason is None
    assert result.se == pytest.approx(expected_se)
    assert (result.lb, result.ub) == pytest.approx(expected_bounds)
    assert result.p_value == pytest.approx(expected_p)
    assert result.valid == 40


def test_centered_bootstrap_reports_genuinely_unrepresentable_spread():
    from increment.estimation.targeting import _centered_bootstrap

    result = _centered_bootstrap(0.0, [-1.7e308, 1.7e308], 0.8, scale=1.0, scales=[1.0, 1.0])
    assert result.reason == "estimation.targeting.nonfinite_statistic"
    assert result.se is result.p_value is result.lb is result.ub is None
    assert result.valid == 2


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
def test_bootstrap_equal_nonbinary_cluster_means_have_zero_spread(weighting):
    from increment.estimation.targeting import (
        _centered_bootstrap,
        _cluster_draw,
        _cluster_mean,
        _cluster_population,
        _target_weights,
    )

    ids = np.repeat(np.arange(3), [2, 4, 6])
    y = np.tile([0.0, 0.2], 6)
    population = _cluster_population(ids, np.zeros(12), stratified=False)
    rng = np.random.default_rng(17)
    samples = []
    for _ in range(39):
        rows, instances = _cluster_draw(population, rng)
        result = _cluster_mean(y[rows], _target_weights(instances, weighting), instances)
        assert result.value == 0.1
        assert result.reason == "estimation.targeting.degenerate_cluster_variance"
        samples.append(result.value)
    uncertainty = _centered_bootstrap(0.1, samples, 0.05, scale=1.0, scales=[1.0] * 39)
    assert uncertainty.reason == "estimation.targeting.bootstrap_zero_variance"
    assert uncertainty.valid == 39


@pytest.mark.parametrize("stage", ["direct", "inner", "outer"])
@pytest.mark.parametrize("partial", [False, True])
def test_cluster_deployment_overlap_requires_complete_retained_rosters(stage, partial):
    from increment.estimation._adjust.overlap import IdentificationError
    from increment.estimation.crossfit import outer_split

    fixture = _c08_fixture(mixed=True)
    ids = fixture["cluster_ids"]
    if stage == "direct":
        population = _content_hash_holdout(fixture["unit_ids"], cluster_ids=ids)
    else:
        outer = outer_split(
            fixture["unit_ids"],
            test_size=0.5,
            seed=17,
            stratify=fixture["d"],
            cluster_ids=ids,
        )
        population = outer if stage == "outer" else ~outer
    rejected_cluster = ids[population][0]
    rejected_unit = fixture["unit_ids"][population][0]

    def trimmed_score(y, d, X, unit_ids, cluster_ids):
        kept = unit_ids != rejected_unit if partial else cluster_ids != rejected_cluster
        return y.copy(), kept

    options = {
        "interact": [SPEND],
        "deploy_grain": "cluster",
        "psi_fn": trimmed_score,
        "arm_summary": "score",
        "n_groups": 2,
        "alpha": 0.4,
        "bootstrap_repetitions": 2,
    }

    def run():
        if stage == "direct":
            return targeting_rule_arrays(**fixture, **options, fraction=0.5)
        return select_targeting_rule_arrays(
            **fixture,
            **options,
            fractions=(0.5,),
            cost_per_treated=0,
            n_folds=2,
            seed=17,
        )

    if partial:
        with pytest.raises(IdentificationError) as caught:
            run()
        assert caught.value.code == "estimation.targeting.overlap.partial_cluster"
    else:
        result = run()
        expected_clusters = np.unique(ids[population]).size - 1
        if stage == "inner":
            assert result.n_clusters == expected_clusters
            assert result.rule.validation.n_train == int(population.sum()) - 6
        else:
            rule = result if stage == "direct" else result.rule
            assert rule.n_clusters == expected_clusters
            assert rule.validation.n_holdout == int(population.sum()) - 6
            assert rule.population == "overlap_subpopulation"


@pytest.mark.parametrize("weighting", ["member_count", "equal"])
@pytest.mark.parametrize("repeated", [False, True])
def test_tied_bootstrap_multiset_permutations_preserve_policy_and_inner_value(
    weighting, repeated, monkeypatch
):
    from increment.estimation import targeting

    chosen = (0, 1, 2, 2) if repeated else (0, 1, 2, 3)
    patterns = (chosen, chosen[::-1])
    fixture: dict = _c08_fixture(mixed=True) | {"cols": {}}

    def run(sequence):
        # Draw whole clusters by population position, so the two runs see the
        # same multiset of draws in opposite orders, in whatever index space
        # the inner and outer callers use.
        cycle = itertools.cycle(sequence)

        def draw(population, rng):
            parts = [population.members[g % len(population.members)] for g in next(cycle)]
            return (
                np.concatenate(parts),
                np.concatenate([np.full(part.size, i) for i, part in enumerate(parts)]),
            )

        monkeypatch.setattr(targeting, "_cluster_draw", draw)
        return select_targeting_rule_arrays(
            **fixture,
            interact=[],
            cluster_weight=weighting,
            fractions=(0.5,),
            cost_per_treated=0.0,
            n_folds=2,
            seed=17,
            n_groups=2,
            intervention_grain="cluster",
            alpha=0.8,
            bootstrap_repetitions=2,
            bootstrap_seed=0,
            include_evaluation_population=True,
        )

    first, second = run(patterns), run(patterns[::-1])
    # Every replicate in both stages resampled: the draw order cannot matter.
    assert first.inner[0].bootstrap_valid_repetitions == 2
    assert first.rule.bootstrap_valid_repetitions == 2
    assert first.selected_fraction == second.selected_fraction
    assert first.model_dump() == second.model_dump()
    # A constant score ties every cluster, so the locked policy is the
    # canonical-ID prefix and does not depend on the draw order either.
    ids = np.asarray(first.rule.validation.evaluation_population.cluster_ids)
    labels = sorted(set(ids.tolist()))
    expected = np.isin(ids, labels[: len(labels) // 2])
    np.testing.assert_array_equal(first.rule.predict({}, cluster_ids=ids), expected)
    np.testing.assert_array_equal(
        second.rule.predict({}, cluster_ids=ids), first.rule.predict({}, cluster_ids=ids)
    )


@pytest.mark.parametrize("fraction", [0.4, 1.0])
@pytest.mark.parametrize("field", ["threshold", "score_cutoff"])
@pytest.mark.parametrize("mutation", ["different", "null"])
def test_policy_json_rejects_disagreement_with_published_cutoff(fraction, field, mutation):
    import json

    from increment.estimation.targeting import TargetingRule

    rule = targeting_rule_arrays(
        **_step_fixture(),
        interact=[SPEND],
        n_groups=4,
        fraction=fraction,
    )
    assert rule.threshold is not None
    payload = json.loads(rule.model_dump_json())
    payload[field] = None if mutation == "null" else payload[field] + 1
    with pytest.raises(InvalidRequestError) as caught:
        TargetingRule.model_validate_json(json.dumps(payload))
    assert caught.value.code == "estimation.targeting.policy_metadata"


def test_failed_gate_policy_keeps_hidden_cutoff_through_json():
    from increment.estimation.targeting import TargetingRule

    fixture = _step_fixture()
    rule = targeting_rule_arrays(**fixture, interact=[SPEND], n_groups=4, fraction=0.4, alpha=0.01)
    assert not rule.validation.passed
    restored = TargetingRule.model_validate_json(rule.model_dump_json())
    assert restored.threshold is None and restored.score_cutoff is not None
    assert restored.recommendation == "simple"
    # The unpublished cutoff is still the deployment rule predict() applies.
    np.testing.assert_array_equal(
        restored.predict(fixture["cols"]),
        restored.score_state.score(fixture["cols"]) >= restored.score_cutoff,
    )


@pytest.mark.parametrize("selection", [False, True])
@pytest.mark.parametrize("grain", ["cluster", "unit"])
def test_constant_policy_predicts_from_roster_without_dummy_columns(selection, grain):
    from increment.estimation.targeting import TargetingRule

    fixture: dict = _c08_fixture() | {"cols": {}}
    if grain == "unit":
        fixture.pop("cluster_ids")
    options = {
        "interact": [],
        "deploy_grain": grain,
        "n_groups": 2,
        "bootstrap_repetitions": 2,
    }
    if selection:
        rule = select_targeting_rule_arrays(
            **fixture,
            **options,
            fractions=(0.5,),
            cost_per_treated=0,
            n_folds=2,
            seed=17,
        ).rule
    else:
        rule = targeting_rule_arrays(**fixture, **options, fraction=0.5)
    restored = TargetingRule.model_validate_json(rule.model_dump_json())
    roster = np.array(["z", "a", "z"])
    expected = [False, True, False] if grain == "cluster" else [True, True, True]
    np.testing.assert_array_equal(restored.predict({}, cluster_ids=roster), expected)
    assert rule.validation.n_holdout > 0
    assert (rule.n_clusters is None) == (grain == "unit")


def _c01_five_row_pattern(replication=1):
    """Repeat every member of each five-row pattern, retaining its cluster identity."""
    groups, positions = [], []
    for g in range(40):
        pattern = np.tile(np.arange(5), replication)
        groups.extend([g] * pattern.size)
        positions.extend(pattern)
    g, j = np.asarray(groups), np.asarray(positions)
    x = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])[j]
    d = (g % 2).astype(float)
    baseline = 0.4 * np.sin(g) + 0.2 * x
    tau = 0.5 + 0.3 * (g % 3) + (0.6 + 0.04 * (g % 7)) * x
    y = baseline + d * tau + np.asarray([-0.2, 0.1, 0.3, -0.1, -0.1])[j]
    return {
        "y": y,
        "d": d,
        "cols": {"spend": x, "profile": x * (1 + 0.03 * g) + 0.2 * np.sin(g)},
        "unit_ids": np.array([f"g{c}-u{i}" for i, c in enumerate(g)]),
        "cluster_ids": np.array([f"g{c}" for c in g]),
    }


def _c01_frozen_ipw(y, d, X, unit_ids, cluster_ids):
    # Known p=.5 and fixed zero nuisance means, independent of the evaluation rows.
    return 2 * (2 * d - 1) * y, np.ones(y.size, dtype=bool)


def _c01_disjoint_arm_oracle(base, weights):
    ids, y, d = base["cluster_ids"], base["y"], base["d"]
    means, variances, counts = [], [], []
    for arm in (0, 1):
        mask = d == arm
        mean = np.average(y[mask], weights=weights[mask])
        arm_ids = np.unique(ids[mask])
        contributions = (
            np.array(
                [np.sum(weights[ids == label] * (y[ids == label] - mean)) for label in arm_ids]
            )
            / weights[mask].sum()
        )
        means.append(mean)
        counts.append(arm_ids.size)
        variances.append(arm_ids.size / (arm_ids.size - 1) * np.sum(contributions**2))
    expected_df = sum(variances) ** 2 / sum(
        variance**2 / (count - 1) for variance, count in zip(variances, counts, strict=True)
    )
    return means[1] - means[0], np.sqrt(sum(variances)), expected_df


def _assert_c01_overlap_profile_oracle(base, weights):
    from increment.estimation.targeting import _cluster_contrast

    ids = base["cluster_ids"]
    score, x = base["cols"]["spend"], base["cols"]["profile"]
    most, least = score > 0, score <= 0
    high = np.average(x[most], weights=weights[most])
    low = np.average(x[least], weights=weights[least])
    signed = np.array(
        [
            np.sum(weights[most & (ids == label)] * (x[most & (ids == label)] - high))
            / weights[most].sum()
            - np.sum(weights[least & (ids == label)] * (x[least & (ids == label)] - low))
            / weights[least].sum()
            for label in np.unique(ids)
        ]
    )
    overlap = _cluster_contrast(
        x[most], x[least], weights[most], weights[least], ids[most], ids[least]
    )
    assert overlap.value == pytest.approx(high - low)
    assert overlap.se == pytest.approx(np.sqrt(40 / 39 * np.sum(signed**2)))
    assert overlap.se is not None and overlap.se > 0
    assert overlap.df == 39
    assert overlap.reason is None


def _assert_c01_pattern_replication(base, repeated, factor):
    for label in np.unique(base["cluster_ids"]):
        before = base["cluster_ids"] == label
        after = repeated["cluster_ids"] == label
        assert before.sum() == 5 and after.sum() == 5 * factor
        for name in ("y", "d"):
            np.testing.assert_array_equal(
                repeated[name][after], np.tile(base[name][before], factor)
            )
        for name in base["cols"]:
            np.testing.assert_array_equal(
                repeated["cols"][name][after], np.tile(base["cols"][name][before], factor)
            )


@pytest.mark.slow
@pytest.mark.parametrize("weighting", ["member_count", "equal"])
@pytest.mark.parametrize("replication", [1, 5, 20, 100])
def test_cluster_complete_five_row_pattern_fixed_validation(weighting, replication):
    from increment.estimation.targeting import _Holdout

    def evaluate(factor):
        data = _c01_five_row_pattern(factor)
        hold = _Holdout(
            n_train=data["y"].size,
            score=data["cols"]["spend"],
            covariates=(Covariate(name="profile"),),
            psi_fn=_c01_frozen_ipw,
            cluster_weight=weighting,
            **data,
        )
        psi = 2 * (2 * hold.d - 1) * hold.y
        result = _validation(
            hold,
            psi,
            n_groups=2,
            alpha=0.05,
            arm_summary="welch",
            bootstrap_seed=57,
            bootstrap_repetitions=99,
        )
        return data, result

    base, original = evaluate(1)
    data, repeated = evaluate(replication)
    ids, y = base["cluster_ids"], base["y"]
    labels, inverse, sizes = np.unique(ids, return_inverse=True, return_counts=True)
    weights = np.ones(y.size) if weighting == "member_count" else 1 / sizes[inverse]
    expected_effect, expected_se, expected_df = _c01_disjoint_arm_oracle(base, weights)
    assert original.holdout_ate is not None and repeated.holdout_ate is not None
    assert original.holdout_ate.value == pytest.approx(expected_effect)
    assert original.holdout_ate_se == pytest.approx(expected_se)
    assert original.reference_df == pytest.approx(expected_df)
    assert repeated.holdout_ate.value == pytest.approx(original.holdout_ate.value)
    assert repeated.holdout_ate_se == pytest.approx(expected_se)
    assert repeated.reference_df == pytest.approx(expected_df)
    assert original.n_clusters == repeated.n_clusters == labels.size == 40
    assert repeated.n_holdout == original.n_holdout * replication
    assert repeated.n_train == original.n_train * replication
    _assert_c01_pattern_replication(base, data, replication)
    for left, right in zip(
        (original.autoc, original.qini, *original.groups, *original.clan),
        (repeated.autoc, repeated.qini, *repeated.groups, *repeated.clan),
        strict=True,
    ):
        assert left.se is not None and left.se > 0
        assert left.lb is not None and left.ub is not None
        assert left.unavailable_reason is right.unavailable_reason is None
        for field in ("se", "lb", "ub"):
            assert getattr(right, field) == pytest.approx(getattr(left, field))
        assert left.n_clusters == right.n_clusters
        assert left.cluster_weight == right.cluster_weight == weighting
        assert left.bootstrap_valid_repetitions == right.bootstrap_valid_repetitions == 99
    for left, right in ((original.autoc, repeated.autoc), (original.qini, repeated.qini)):
        assert right.estimate == pytest.approx(left.estimate)
        assert right.p_value == left.p_value
    for left, right in zip(original.groups, repeated.groups, strict=True):
        assert right.effect == pytest.approx(left.effect)
        assert right.mean_score == pytest.approx(left.mean_score)
        assert right.n == left.n * replication
    for left, right in zip(original.clan, repeated.clan, strict=True):
        assert right.diff == pytest.approx(left.diff)

    # Both CLAN groups span the same clusters; square their joint signed influence.
    _assert_c01_overlap_profile_oracle(base, weights)


def test_evaluation_population_is_opt_in_immutable_and_json_roundtrips():
    fixture = _signal_fixture()
    fixture["unit_ids"] = fixture["unit_ids"].copy()
    default = validate_cate_arrays(**fixture, interact=[SPEND])
    assert default.evaluation_population is None
    unit_ids = fixture["unit_ids"]
    result = validate_cate_arrays(**fixture, interact=[SPEND], include_evaluation_population=True)
    population = result.evaluation_population
    assert population is not None
    assert population.split == "honest"
    assert len(population.unit_ids) == result.n_holdout
    restored = type(population).model_validate_json(population.model_dump_json())
    assert restored == population
    assert isinstance(unit_ids, np.ndarray)
    unit_ids[:] = "mutated-after-call"
    assert not any(value.startswith("mutated") for value in population.unit_ids)


def test_selection_evaluation_population_is_outer_only():
    result, populations = (
        TestSelectTargetingRuleArrays._selection_with_different_inner_outer_retention(
            include_evaluation_population=True
        )
    )
    snapshot = result.rule.validation.evaluation_population
    assert snapshot is not None
    assert snapshot.unit_ids == populations[1]
    assert set(snapshot.unit_ids).isdisjoint(populations[0])
    assert result.population == "overlap_subpopulation"
    assert snapshot.retention == "all"
    assert snapshot.overlap is None
    assert snapshot.split == "outer"
    assert snapshot.seed == 11
    assert TargetingSelection.model_validate_json(result.model_dump_json()) == result


def test_evaluation_population_preserves_post_trim_equal_cluster_weights():
    fixture = _signal_fixture()
    cluster_ids = np.repeat([f"c{i}" for i in range(20)], 4)
    held = _content_hash_holdout(fixture["unit_ids"], cluster_ids=cluster_ids)
    retained = held & (np.arange(cluster_ids.size) % 3 != 0)

    def score(y, d, X, unit_ids, clusters):
        kept = np.array([int(value[1:]) % 3 != 0 for value in unit_ids])
        return y + d + X[:, 0], kept

    result = validate_cate_arrays(
        **fixture,
        cluster_ids=cluster_ids,
        cluster_weight="equal",
        interact=[SPEND],
        adjustment=("spend",),
        psi_fn=score,
        arm_summary="score",
        n_groups=2,
        bootstrap_repetitions=99,
        include_evaluation_population=True,
    )
    snapshot = result.evaluation_population
    assert snapshot is not None
    assert snapshot.unit_ids == tuple(fixture["unit_ids"][retained])
    assert snapshot.cluster_ids == tuple(cluster_ids[retained])
    labels, counts = np.unique(cluster_ids[retained], return_counts=True)
    assert len(set(counts)) > 1
    expected = np.array(
        [1.0 / counts[np.flatnonzero(labels == label)[0]] for label in cluster_ids[retained]]
    )
    np.testing.assert_allclose(snapshot.base_weights, expected)
    assert snapshot.retention == "overlap_trimmed"
    assert snapshot.overlap == "overlap_subpopulation"
    assert result.holdout_ate is not None
    scores = (fixture["y"] + fixture["d"] + fixture["cols"]["spend"])[retained]
    assert result.holdout_ate.value == pytest.approx(np.average(scores, weights=expected))


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_weights": (1.0,)},
        {"unit_ids": ("u0", "u0")},
        {"base_weights": (float("nan"), 1.0)},
        {"weighting": "equal"},
        {"split": "outer"},
        {"retention": "overlap_trimmed"},
    ],
)
def test_evaluation_population_rejects_inconsistent_provenance(overrides):
    from increment.errors import InvalidRequestError
    from increment.estimation.targeting import CateEvaluationPopulation

    values = {
        "unit_ids": ("u0", "u1"),
        "base_weights": (1.0, 1.0),
        "weighting": "member_count",
        "split": "honest",
    }
    with pytest.raises(InvalidRequestError) as caught:
        CateEvaluationPopulation.model_validate({**values, **overrides})
    assert caught.value.code == "estimation.targeting.evaluation_population_invalid"


@pytest.mark.parametrize(
    ("weighting", "cluster_ids", "base_weights"),
    [
        ("member_count", ("c0", "c0", "c1"), (2.0, 1.0, 1.0)),
        ("equal", ("c0", "c0", "c1"), (1.0, 1.0, 1.0)),
    ],
)
def test_evaluation_population_rejects_weights_that_disagree_with_weighting(
    weighting, cluster_ids, base_weights
):
    from increment.estimation.targeting import CateEvaluationPopulation

    with pytest.raises(InvalidRequestError) as caught:
        CateEvaluationPopulation(
            unit_ids=("u0", "u1", "u2"),
            cluster_ids=cluster_ids,
            base_weights=base_weights,
            weighting=weighting,
            split="honest",
        )
    assert caught.value.code == "estimation.targeting.evaluation_population_invalid"


def test_selection_rejects_outer_snapshot_with_different_seed():
    result, _ = TestSelectTargetingRuleArrays._selection_with_different_inner_outer_retention(
        include_evaluation_population=True
    )
    payload = result.model_dump()
    assert payload["rule"]["validation"]["evaluation_population"] is not None
    payload["rule"]["validation"]["evaluation_population"]["seed"] = result.seed + 1
    with pytest.raises(InvalidRequestError) as caught:
        type(result).model_validate(payload)
    assert caught.value.code == "estimation.targeting.policy_metadata"


def test_validation_rejects_a_snapshot_for_a_different_population():
    from increment.errors import InvalidRequestError

    result = validate_cate_arrays(
        **_signal_fixture(), interact=[SPEND], include_evaluation_population=True
    )
    payload = result.model_dump()
    payload["n_holdout"] += 1
    with pytest.raises(InvalidRequestError) as caught:
        type(result).model_validate(payload)
    assert caught.value.code == "estimation.targeting.evaluation_population_invalid"


# Categorical observational adjustment through the built-in doubly robust
# score: the raw string column must reproduce hand-built modal-reference
# dummies on every honest-split entry point, clustered or not.


def _categorical_fixture(*, clustered: bool) -> tuple[dict, dict]:
    from tests.categorical_cases import DUMMY_COLUMNS, LEVELS, categorical_units

    units = categorical_units(400, seed=13)
    d = (units["variant"] == "T").astype(float)
    y = units["revenue"] + 0.6 * units["spend"] * d
    base = {"y": y, "d": d, "unit_ids": units["user_id"]}
    if clustered:
        base["cluster_ids"] = np.array([f"c{i % 50}" for i in range(400)])
    raw = {**base, "cols": {"spend": units["spend"], "region": units["region"]}}
    dummies = {**base, "cols": {"spend": units["spend"]}}
    for level, column in zip(LEVELS[1:], DUMMY_COLUMNS, strict=True):
        dummies["cols"][column] = (units["region"] == level).astype(float)
    return raw, dummies


def _dr_plan():
    return functools.partial(
        _dr_psi,
        propensity_learner=LogisticPropensity,
        outcome_learner=RidgeOutcome,
        folds=4,
        seed=3,
        gate=IdentificationGate(overlap="trim"),
    )


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize(
    "caller", ["validate_cate_arrays", "targeting_rule_arrays", "select_targeting_rule_arrays"]
)
def test_array_entry_points_categorical_adjustment_matches_dummies(caller, clustered):
    import warnings

    from increment.errors import IncrementWarning
    from tests.categorical_cases import DUMMY_COLUMNS, assert_rows_match

    raw, dummies = _categorical_fixture(clustered=clustered)
    options = {
        "interact": [SPEND],
        "psi_fn": _dr_plan(),
        "arm_summary": "score",
        "bootstrap_repetitions": 49,
    }
    oracle = _call_array_entry_point(
        caller, dummies, adjustment=("spend", *DUMMY_COLUMNS), **options
    ).model_dump()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IncrementWarning)
        actual = _call_array_entry_point(
            caller, raw, adjustment=("spend", "region"), **options
        ).model_dump()
    assert all(
        getattr(item.message, "code", None) == "estimation.targeting.unseen_level_advisory"
        for item in caught
    )
    assert_rows_match(oracle, actual, skip=("required_columns",))


@pytest.mark.parametrize("clustered", [False, True])
def test_validation_discloses_a_holdout_level_its_nuisance_fits_never_saw(clustered):
    """A level one held-out unit alone carries is absent from every nuisance
    fit that scores it -- the propensity and both arm outcome models,
    cross-fitted inside the holdout or frozen on the training half of a
    clustered source -- so each reads it as the reference level. The coded
    advisory names that one row and those three fits over the holdout;
    the same fixture without the lone level raises nothing."""
    import warnings

    def advisories(fixture):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_cate_arrays(
                **fixture,
                interact=[SPEND],
                adjustment=("spend", "region"),
                psi_fn=_dr_plan(),
                arm_summary="score",
                bootstrap_repetitions=49,
            )
        return [
            w.message
            for w in caught
            if getattr(w.message, "code", None) == "estimation.targeting.unseen_level_advisory"
        ]

    raw, _ = _categorical_fixture(clustered=clustered)
    assert advisories(raw) == []

    hold = _content_hash_holdout(raw["unit_ids"], raw.get("cluster_ids"))
    region = raw["cols"]["region"].astype(object)
    region[int(np.flatnonzero(hold)[0])] = "island"
    raw["cols"]["region"] = region
    (advisory,) = advisories(raw)
    assert advisory.context["unseen"] == (("region", "island", 1, 3),)
    assert advisory.context["n"] == int(hold.sum())
