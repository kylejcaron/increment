"""Unit tests for the ``absorb_factor`` narwhals adapter."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from increment import absorb_factor
from increment.errors import IncrementRuntimeWarning
from increment.estimation.absorption import absorb_one_way
from increment.estimation.armstats import centered_row_from_raw_sums
from tests.warning_codes import warning_codes


def _cell(country: str, group_id: str, n: int, sum_y: float, sum_y2: float) -> dict:
    """One centered moment row, from the raw sums the expectations are written in."""
    return centered_row_from_raw_sums(
        {"country": country, "group_id": group_id, "n": n, "sum_y": sum_y, "sum_y2": sum_y2}
    )


# Three factor levels, control + treatment, deliberately unequal cell sizes.
_ROWS = [
    _cell("US", "control", 10, 10.0, 20.0),
    _cell("US", "treatment", 12, 18.0, 36.0),
    _cell("CA", "control", 20, 40.0, 100.0),
    _cell("CA", "treatment", 18, 45.0, 130.0),
    _cell("MX", "control", 5, 2.5, 6.5),
    _cell("MX", "treatment", 6, 4.8, 10.0),
]


def _table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows)


def _offset_rows(offset: float) -> list[dict]:
    """Eight levels of a real one-way DGP, every y shifted by *offset*."""
    rng = np.random.default_rng(4)
    out = []
    for level in range(8):
        b = rng.normal(0.0, 2.0)
        for group, eff in (("control", 0.0), ("treatment", 1.5)):
            y = offset + b + eff + rng.normal(0.0, 1.0, 60)
            ref = float(y.mean())
            out.append(
                {
                    "country": f"c{level}",
                    "group_id": group,
                    "n": len(y),
                    "ref_y": ref,
                    "cy1": float(np.sum(y - ref)),
                    "cy2": float(np.sum((y - ref) ** 2)),
                }
            )
    return out


def _many_level_rows(n_levels: int) -> list[dict]:
    """*n_levels* factor levels, control + treatment, identical cells --
    shape only matters for the level-count guard, not the estimate."""
    rows = []
    for i in range(n_levels):
        level = f"L{i:03d}"
        rows.append(_cell(level, "control", 10, 20.0, 42.0))
        rows.append(_cell(level, "treatment", 10, 22.0, 50.0))
    return rows


@pytest.mark.filterwarnings("ignore:factor .* levels survived absorption, below 40:RuntimeWarning")
def test_effect_and_se_are_invariant_to_a_large_location_shift():
    """The one-way model absorbs a location shift into its intercept, and
    the adapter re-references every cell against one pooled mean rather
    than recovering raw sums - so a 1e9 offset must not move the contrast.
    The raw-sum wire format lost this entirely: sum(y**2) at mu=1e9 carries
    no within-cell dispersion at all.
    """
    base = absorb_factor(_table(_offset_rows(0.0)), factor="country", control_group="control")
    shifted = absorb_factor(_table(_offset_rows(1e9)), factor="country", control_group="control")
    assert shifted.effect == pytest.approx(base.effect, rel=1e-9)
    assert shifted.se == pytest.approx(base.se, rel=1e-9)
    # icc and the unadjusted SE are differences of mean-square terms, one
    # cancellation further out than the contrast itself.
    assert shifted.icc == pytest.approx(base.icc, rel=1e-7)
    assert shifted.se_unadjusted == pytest.approx(base.se_unadjusted, rel=1e-7)


class TestHappyPath:
    @pytest.mark.filterwarnings(
        "ignore:factor .* levels survived absorption, below 40:RuntimeWarning"
    )
    def test_matches_hand_built_absorb_one_way(self):
        """Pivoting the group_summary-shaped table must reproduce the
        parallel-array call to absorb_one_way exactly (levels sorted the
        same way the adapter sorts them: CA, MX, US by string)."""
        result = absorb_factor(_table(_ROWS), factor="country", control_group="control")
        # Levels sorted by str(level): "CA" < "MX" < "US".
        n_c = [20.0, 5.0, 10.0]
        s_c = [40.0, 2.5, 10.0]
        q_c = [100.0, 6.5, 20.0]
        n_t = [18.0, 6.0, 12.0]
        s_t = [45.0, 4.8, 18.0]
        q_t = [130.0, 10.0, 36.0]
        expected = absorb_one_way(n_c, s_c, q_c, n_t, s_t, q_t)
        assert result.effect == pytest.approx(expected.effect)
        assert result.se == pytest.approx(expected.se)

    @pytest.mark.filterwarnings(
        "ignore:factor .* levels survived absorption, below 40:RuntimeWarning"
    )
    def test_arm_order_does_not_matter(self):
        """control_group picks the control arm regardless of row order or
        which group_id happens to sort first."""
        rows = list(reversed(_ROWS))
        result = absorb_factor(_table(rows), factor="country", control_group="control")
        result2 = absorb_factor(_table(_ROWS), factor="country", control_group="control")
        assert result.effect == pytest.approx(result2.effect)


class TestGuards:
    def test_missing_required_column_raises(self):
        from increment.errors import InvalidRequestError

        rows = [{k: v for k, v in r.items() if k != "cy2"} for r in _ROWS]
        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_factor(_table(rows), factor="country", control_group="control")
        assert exc_info.value.code == "absorption.summary_missing_columns"
        assert exc_info.value.context["missing"] == ("cy2",)

    def test_non_eager_frame_raises(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_factor(object(), factor="country", control_group="control")  # ty: ignore[invalid-argument-type]
        assert exc_info.value.code == "absorption.summary_eager_dataframe"

    def test_control_group_absent_raises(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_factor(_table(_ROWS), factor="country", control_group="nonexistent")
        assert exc_info.value.code == "absorption.control_group_present"
        assert exc_info.value.context["control_group"] == "nonexistent"

    def test_more_than_two_arms_raises(self):
        from increment.errors import InvalidRequestError

        rows = _ROWS + [_cell("US", "treatment_b", 8, 12.0, 24.0)]
        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_factor(_table(rows), factor="country", control_group="control")
        assert exc_info.value.code == "absorption.absorb_factor_needs"

    def test_duplicate_level_arm_rows_refused(self):
        """A duplicated (level, arm) cell must refuse, not silently
        last-row-wins: the audit measured a duplicate row moving the
        effect 0.46 -> 0.97 with zero signal. One row per cell is the
        input contract; a duplicate is an upstream GROUP BY mistake."""
        from increment.errors import InvalidRequestError

        rows = _ROWS + [_cell("US", "treatment", 12, 180.0, 3600.0)]
        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_factor(_table(rows), factor="country", control_group="control")
        assert exc_info.value.code == "absorption.summary_duplicate_rows"
        assert (
            "US" in exc_info.value.context["named"]  # ty: ignore[unsupported-operator]
            and "treatment" in exc_info.value.context["named"]  # ty: ignore[unsupported-operator]
        )


class TestLevelCountGuard:
    """absorb_factor's level-clustered sandwich (t_(K-2), CR1) has
    the same small-K fragility as any other cluster-robust path -- warn
    below 40 surviving levels, mirroring check_total_clusters' warn tier."""

    def test_few_levels_warns_of_sandwich_fragility(self):
        with pytest.warns(IncrementRuntimeWarning) as rec:
            absorb_factor(_table(_offset_rows(0.0)), factor="country", control_group="control")
        assert "absorption.factor_levels_below_sandwich_floor" in warning_codes(rec)

    def test_forty_levels_does_not_warn(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            absorb_factor(_table(_many_level_rows(40)), factor="country", control_group="control")

    def test_thirty_nine_levels_warns(self):
        with pytest.warns(IncrementRuntimeWarning) as rec:
            absorb_factor(_table(_many_level_rows(39)), factor="country", control_group="control")
        assert "absorption.factor_levels_below_sandwich_floor" in warning_codes(rec)


class TestLevelPresentInOnlyOneArm:
    @pytest.mark.filterwarnings(
        "ignore:factor .* levels survived absorption, below 40:RuntimeWarning"
    )
    def test_level_missing_from_one_arm_treated_as_zero_cell(self):
        rows = _ROWS + [_cell("BR", "control", 4, 4.0, 8.0)]
        result = absorb_factor(_table(rows), factor="country", control_group="control")
        # Levels sorted by str(level): "BR" < "CA" < "MX" < "US".
        n_c = [4.0, 20.0, 5.0, 10.0]
        s_c = [4.0, 40.0, 2.5, 10.0]
        q_c = [8.0, 100.0, 6.5, 20.0]
        n_t = [0.0, 18.0, 6.0, 12.0]
        s_t = [0.0, 45.0, 4.8, 18.0]
        q_t = [0.0, 130.0, 10.0, 36.0]
        expected = absorb_one_way(n_c, s_c, q_c, n_t, s_t, q_t)
        assert result.effect == pytest.approx(expected.effect)
        assert result.se == pytest.approx(expected.se)
