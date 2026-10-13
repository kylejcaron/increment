"""Multiplicity, request refusals, design refusals, serialization, and public exports."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

import increment
from increment.errors import CodedError, UnsupportedRequestError
from increment.overlap import (
    CONTRASTS,
    OverlapCells,
    OverlapDesign,
    OverlapExperiment,
    OverlapReadout,
    estimate_overlap_effects,
)
from tests.overlap.fixtures import DESIGN, from_frame, generated_cells, unit_frame

MU = {(0, 0): 1.0, (1, 0): 1.3, (0, 1): 1.0, (1, 1): 1.9}
SIZES = {(0, 0): 200, (1, 0): 200, (0, 1): 70, (1, 1): 60}


def _cells(**metrics: float) -> OverlapCells:
    rng = np.random.default_rng(3)
    frame = unit_frame(generated_cells(rng, MU, SIZES))
    for name, shift in metrics.items():
        frame[name] = frame["revenue"] + shift + rng.standard_normal(len(frame))
    return from_frame(frame, metrics={"revenue": "revenue", **{m: m for m in metrics}})


def test_primary_rows_form_one_bonferroni_family_with_simultaneous_intervals() -> None:
    cells = _cells(orders=0.0)
    declared = ("interaction", "joint_launch", "first_only_launch")
    readout = estimate_overlap_effects(
        cells, contrasts=declared, roles={"revenue": "primary", "orders": "primary"}, alpha=0.05
    )
    unadjusted = estimate_overlap_effects(cells, contrasts=declared, alpha=0.05)
    for row, raw in zip(readout.rows, unadjusted.rows, strict=True):
        assert row.multiplicity_status == "bonferroni_fwer"
        assert row.p_value is not None and row.lb is not None and row.ub is not None
        assert raw.lb is not None and raw.ub is not None
        assert row.family_size == 6
        assert row.test_level == pytest.approx(0.05 / 6)
        assert row.adjusted_p_value == pytest.approx(min(1.0, 6 * row.p_value))
        assert row.significant == (row.p_value <= 0.05 / 6)
        assert row.ub - row.lb > raw.ub - raw.lb
        assert raw.multiplicity_status == "exploratory_unadjusted"
        assert (raw.family_size, raw.test_level) == (1, 0.05)


def test_secondary_rows_use_bh_at_q_and_keep_marginal_intervals() -> None:
    cells = _cells(orders=0.0, sessions=0.0)
    readout = estimate_overlap_effects(
        cells,
        contrasts=CONTRASTS,
        roles={"revenue": "primary", "orders": "secondary", "sessions": "secondary"},
        q=0.10,
    )
    secondary = [r for r in readout.rows if r.role == "secondary"]
    assert {r.family_size for r in secondary} == {2 * len(CONTRASTS)}
    ps = np.array([r.p_value for r in secondary])
    m = len(ps)
    order = np.argsort(ps)
    expected = np.minimum.accumulate((ps[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(expected, 1.0)
    assert [r.adjusted_p_value for r in secondary] == pytest.approx(adjusted.tolist())
    assert [r.significant for r in secondary] == [bool(a <= 0.10) for a in adjusted]
    assert all(r.test_level == 0.05 and r.multiplicity_status == "bh_fdr" for r in secondary)


def test_unavailable_members_stay_in_the_family_without_a_verdict() -> None:
    frame = unit_frame({(0, 0): [1.0, 2.0, 3.0], (1, 0): [4.0, 6.0, 5.0], (0, 1): [1.0, 2.0]})
    cells = from_frame(frame, DESIGN)
    readout = estimate_overlap_effects(
        cells, contrasts=("first_only_launch", "joint_launch"), roles={"revenue": "secondary"}
    )
    available, unavailable = readout.rows
    assert available.family_size == unavailable.family_size == 2
    assert unavailable.adjusted_p_value is None and unavailable.significant is None
    assert available.p_value is not None
    assert available.adjusted_p_value == pytest.approx(min(1.0, 2 * available.p_value))


REQUEST_REFUSALS = {
    "overlap.request.contrasts_empty": {"contrasts": ()},
    "overlap.request.contrast_unknown": {"contrasts": ("both_minus_neither",)},
    "overlap.request.contrast_duplicate": {"contrasts": ("interaction", "interaction")},
    "overlap.request.metric_unknown": {"roles": {"latency": "primary"}},
    "overlap.request.role_unknown": {"roles": {"revenue": "tertiary"}},
    "overlap.multiplicity.guardrail_unfinished": {"roles": {"revenue": "guardrail"}},
    "overlap.request.alpha": {"alpha": 1.5},
    "overlap.request.q": {"q": 0},
}


@pytest.mark.parametrize("code", sorted(REQUEST_REFUSALS))
def test_request_refusals_are_coded(code: str) -> None:
    kwargs = {"contrasts": ("interaction",), **REQUEST_REFUSALS[code]}
    with pytest.raises(CodedError) as exc:
        estimate_overlap_effects(_cells(), **kwargs)  # ty: ignore[invalid-argument-type] -- deliberately invalid request values
    assert exc.value.code == code


@pytest.mark.parametrize(
    "code, build",
    [
        (
            "overlap.design.arm_labels",
            lambda: OverlapExperiment(name="a", control="x", treatment="x"),
        ),
        (
            "overlap.design.treatment_probability",
            lambda: OverlapExperiment(name="a", treatment_probability=1.0),
        ),
        (
            "overlap.design.duplicate_experiment",
            lambda: OverlapDesign(
                first=OverlapExperiment(name="a"), second=OverlapExperiment(name="a")
            ),
        ),
        (
            "overlap.design.joint_law_unsupported",
            lambda: OverlapDesign(
                first=OverlapExperiment(name="a"),
                second=OverlapExperiment(name="b"),
                joint_law="mutually_exclusive",
            ),
        ),
    ],
)
def test_design_refusals_are_coded_and_survive_round_trips(code: str, build) -> None:
    with pytest.raises(CodedError) as exc:
        build()
    assert exc.value.code == code
    for restored in (pickle.loads(pickle.dumps(exc.value)), copy.deepcopy(exc.value)):
        assert restored.code == code and str(restored) == str(exc.value)
    if code == "overlap.design.joint_law_unsupported":
        assert isinstance(exc.value, UnsupportedRequestError)


def test_readout_round_trips_through_json_pickle_and_deepcopy() -> None:
    readout = estimate_overlap_effects(
        _cells(orders=0.5), contrasts=CONTRASTS, roles={"revenue": "primary"}
    )
    restored = OverlapReadout.model_validate_json(readout.model_dump_json())
    assert restored == readout
    assert pickle.loads(pickle.dumps(readout)) == readout
    assert copy.deepcopy(readout) == readout
    assert restored.design.second.treatment_probability == 0.25
    assert [r.contrast for r in restored.rows[: len(CONTRASTS)]] == list(CONTRASTS)


def test_assignment_check_compares_joint_cells_with_the_declared_law() -> None:
    readout = estimate_overlap_effects(_cells(), contrasts=("interaction",))
    check = readout.assignment_check
    assert check.df == 3 and check.alpha == 0.001
    assert check.cells == ("control|old", "treatment|old", "control|new", "treatment|new")
    assert check.observed == (200, 200, 70, 60)
    assert check.expected_shares == pytest.approx((0.375, 0.375, 0.125, 0.125))
    expected = np.array(check.expected_shares) * 530
    chi2 = float(((np.array(check.observed) - expected) ** 2 / expected).sum())
    assert check.chi2_stat == pytest.approx(chi2)
    assert not check.mismatch


def test_mismatched_joint_allocation_is_flagged() -> None:
    sizes = {(0, 0): 400, (1, 0): 100, (0, 1): 100, (1, 1): 400}
    frame = unit_frame(generated_cells(np.random.default_rng(1), MU, sizes))
    readout = estimate_overlap_effects(from_frame(frame, DESIGN), contrasts=("interaction",))
    assert readout.assignment_check.mismatch


def test_root_and_results_exports_resolve_to_the_family() -> None:
    from increment import results

    assert increment.estimate_overlap_effects is estimate_overlap_effects
    assert increment.OverlapDesign is OverlapDesign
    assert results.OverlapReadout is OverlapReadout
    for name in (
        "OverlapCells",
        "OverlapExperiment",
        "ExperimentColumns",
        "overlap_contrast_power",
        "overlap_minimum_detectable_effect",
        "simulate_overlap_power",
    ):
        assert name in increment.__all__ and getattr(increment, name) is not None
