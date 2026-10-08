"""Finite independent winsorized-inference oracles; no simulation or warehouse network access."""

import math
import operator
from collections.abc import Callable
from fractions import Fraction
from itertools import product
from typing import cast

import pytest


def _raw(
    control=(1, 2, 3, 9),
    treatment=(1, 2, 3, 5),
    *,
    upper=None,
    quantile=0.75,
    population="assigned",
    study_id="e",
):
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState, WinsorSupport

    return WinsorRawState(
        metric="revenue",
        study_id=study_id,
        population=population,
        missingness="error",
        quantile=quantile,
        inference=WinsorInferenceSpec(method="joint-rank-projection-v1"),
        support=WinsorSupport(lower=0, upper=upper, provenance="External finite oracle support"),
        arms=(RawArm(group_id="C", values=control), RawArm(group_id="T", values=treatment)),
    )


def test_rational_heterogeneous_influence():
    from increment.estimation.winsor import InfluenceArm, influence_variance

    n = 20
    # E[min(U[0,2],6/5)^2]=108/125; m_T^2=441/625.
    arms = {
        "C": InfluenceArm(n, 0.5, 0.5, 1 / 12, 1),
        "T": InfluenceArm(n, 0.5, 21 / 25, 99 / 625, 3 / 5),
    }
    value = influence_variance(arms, "C", "T", 6 / 5, 1 / 4, "log_ratio")
    assert value.fixed == pytest.approx(82 / (147 * n))
    assert value.cutoff == pytest.approx(32 / (147 * n))
    assert value.cross == pytest.approx(16 / (49 * n))
    assert value.total == pytest.approx(54 / (49 * n))
    additive = influence_variance(arms, "C", "T", 6 / 5, 1 / 4, "difference")
    # Direct rational integration of the additive scores, independent of
    # scaling the log variance: C score=-(Y-.5), T score=W-.84+.8(.6-I).
    assert additive.fixed == pytest.approx((Fraction(1, 12) + Fraction(99, 625)) / n)
    assert additive.cutoff == pytest.approx(Fraction(96, 625) / n)
    assert additive.cross == pytest.approx(Fraction(144, 625) / n)


def test_influence_third_arm_and_signed_covariance():
    from increment.estimation.winsor import InfluenceArm, influence_variance

    arms = {
        "C": InfluenceArm(4, 0.2, 0.8, 0.12, 0.6),
        "T": InfluenceArm(8, 0.4, 0.5, 0.09, 0.9),
        "O": InfluenceArm(8, 0.4, 0.6, 0.10, 0.7),
    }
    result = influence_variance(arms, "C", "T", 1, 0.5, "difference")
    # A=-.3; the treatment cross term is negative. The third arm has
    # coefficient a=0 but contributes (.24)^2*.7*.3/8 to cutoff variance.
    expected = (
        (0.12 / 4 + 0.09 / 8),
        (0.12**2 * 0.6 * 0.4 / 4 + 0.24**2 * 0.9 * 0.1 / 8 + 0.24**2 * 0.7 * 0.3 / 8),
        (2 * 0.12 * 0.4 * 0.2 / 4 - 2 * 0.24 * 0.1 * 0.5 / 8),
    )
    assert (result.fixed, result.cutoff, result.cross) == pytest.approx(expected)


@pytest.mark.parametrize("contrast", ["log_ratio", "difference"])
def test_empirical_studentization_matches_rational_centered_scores(contrast):
    from increment.estimation.winsor import influence_studentization
    from increment.winsor import RawArm, WinsorRawState

    payload = _raw((1, 2), (2, 4), quantile=0.5).model_dump()
    payload.pop("allocation")
    payload["arms"] = (*payload["arms"], RawArm(group_id="O", values=(3, 9)).model_dump())
    raw = WinsorRawState.model_validate(payload)
    # Sorted pool 1,2,2,3,4,9 has type-7 cutoff 5/2. Use exact score algebra.
    cutoff, density = Fraction(5, 2), Fraction(1, 2)
    samples = {a.group_id: tuple(map(Fraction, a.values)) for a in raw.arms}
    means = {g: sum(min(y, cutoff) for y in ys) / len(ys) for g, ys in samples.items()}
    cdfs = {g: Fraction(sum(y <= cutoff for y in ys), len(ys)) for g, ys in samples.items()}
    coefficients = {"C": -1 / means["C"], "T": 1 / means["T"], "O": Fraction(0)}
    if contrast == "difference":
        coefficients = {"C": Fraction(-1), "T": Fraction(1), "O": Fraction(0)}
    derivative = sum(coefficients[g] * (1 - cdfs[g]) for g in samples)
    expected = Fraction(0)
    for g, ys in samples.items():
        scores = tuple(
            coefficients[g] * (min(y, cutoff) - means[g])
            + derivative / (3 * density) * (cdfs[g] - int(y <= cutoff))
            for y in ys
        )
        center = sum(scores) / len(scores)
        expected += sum((s - center) ** 2 for s in scores) / (len(scores) * (len(scores) - 1))
    assert influence_studentization(
        raw, "C", "T", density=float(density), contrast=contrast
    ) == pytest.approx(float(expected))


def test_linear_quantile_and_point_near_overflow():
    from increment.estimation.winsor import estimate_winsor_lift

    raw = _raw((1, 2), (1, 2), quantile=0.5).model_dump()
    raw["support"]["lower"] = -1e308
    for arm in raw["arms"]:
        arm["values"] = (-1e308, 1e308)
    from increment.winsor import WinsorRawState

    # Construct only after supplying the signed support declaration. The
    # cutoff lands on the exact midpoint of the widest finite gap; if it
    # overflowed, min(y, cutoff) would leave the arm means summing to zero
    # and no finite point would be reported.
    state = WinsorRawState.model_validate(raw)
    row = estimate_winsor_lift(state, "C", "T")
    assert row.confidence_set is not None
    assert row.confidence_set.point == 0
    assert row.confidence_set.additive_point == 0


def test_linear_quantile_finite_bias_and_empirical_target():
    import numpy as np

    n, q = 100, Fraction(99, 100)
    rank = 1 + (n - 1) * q
    expectation = rank / (n + 1)
    assert expectation == Fraction(9901, 10100)
    assert float(expectation) == pytest.approx(0.9802970297029703)
    assert expectation < q
    # Type 7 is an estimator; the empirical generating distribution's
    # generalized inverse is a different finite-population target.
    sample = np.arange(1, 101)
    assert np.quantile(sample, 0.99, method="linear") == pytest.approx(99.01)
    assert sample[math.ceil(n * q) - 1] == 99


@pytest.mark.parametrize(
    "nt,nc,minimum", [(50, 50, 0.86270), (50, 200, 0.35018), (200, 50, 0.38120)]
)
def test_missing_tail_impossibility(nt, nc, minimum):
    q, alpha = Fraction(99, 100), Fraction(1, 20)
    critical_mass = (1 - q) / Fraction(nt, nt + nc)
    unseen = (1 - critical_mass) ** nt
    forced_unbounded = 1 - alpha / unseen
    assert float(forced_unbounded) >= minimum
    assert forced_unbounded > Fraction(1, 100)
    if nt == nc:
        assert float(unseen) == pytest.approx(0.36416968008711675)


def _simplex_probability(a, b):
    """Independent exact multinomial partition of the ordered simplex.

    Enumerate interval counts, constrain cumulative ranks at every band
    boundary, and sum exact multinomial probabilities. No determinant.
    """
    n = len(a)
    grid = sorted({Fraction(0), Fraction(1), *a, *b})
    answer = Fraction(0)
    for counts in product(range(n + 1), repeat=len(grid) - 1):
        if sum(counts) != n:
            continue
        ordered_cells = [i for i, count in enumerate(counts) for _ in range(count)]
        if any(grid[cell] < a[j] or grid[cell + 1] > b[j] for j, cell in enumerate(ordered_cells)):
            continue
        probability = Fraction(math.factorial(n))
        for i, count in enumerate(counts):
            probability *= (grid[i + 1] - grid[i]) ** count / math.factorial(count)
        answer += probability
    return answer


@pytest.mark.parametrize(
    "a,b",
    [
        ((0, 0.5), (0.5, 1)),
        ((0, 0.25, 0.25), (0.5, 0.5, 1)),
        ((0.125, 0.25, 0.5), (0.5, 0.75, 0.875)),
        ((0, 0, 0), (0.25, 0.5, 1)),
    ],
)
def test_exact_crossing_matches_independent_simplex(a: tuple[float, ...], b: tuple[float, ...]):
    from decimal import Decimal

    from increment.estimation._rank_bands import crossing_probability

    exact = _simplex_probability(tuple(Fraction(x) for x in a), tuple(Fraction(x) for x in b))
    lo, hi = crossing_probability(tuple(map(float, a)), tuple(map(float, b)))
    target = Decimal(exact.numerator) / Decimal(exact.denominator)
    assert lo <= target <= hi
    assert hi - lo < Decimal("1e-60")


def test_band_error_and_extreme_alpha():
    from decimal import Decimal

    from increment.estimation._rank_bands import (
        bonferroni_rank_band,
        crossing_probability,
        simultaneous_rank_band,
    )

    for n in (1, 2, 4):
        band = simultaneous_rank_band(n, 0.025)
        lo, _ = crossing_probability(band.lower, band.upper)
        assert lo >= 1 - Decimal.from_float(0.025)
        baseline = bonferroni_rank_band(n, 0.025)
        baseline_lo, _ = crossing_probability(baseline.lower, baseline.upper)
        assert baseline_lo >= 1 - Decimal.from_float(0.025)
    band = simultaneous_rank_band(2, math.ulp(0.0))
    assert all(x == 0 for x in band.lower)
    assert all(x == 1 for x in band.upper)


def _contains(interval, value):
    return (interval.lower.value is None or Fraction(interval.lower.value) <= value) and (
        interval.upper.value is None or value <= Fraction(interval.upper.value)
    )


@pytest.mark.slow
@pytest.mark.parametrize("quantile", [0.5, 0.9])
def test_joint_finite_support_projection_with_atoms_and_third_arm(quantile):
    from increment.estimation._rank_bands import RankBand
    from increment.estimation.winsor import project_bands
    from increment.winsor import RawArm, WinsorRawState

    base = _raw((0, 1), (1, 2), upper=2, quantile=quantile)
    raw = WinsorRawState.model_validate(
        {
            **base.model_dump(),
            "allocation": (),
            "arms": (*base.arms, RawArm(group_id="O", values=(0, 2, 2))),
        }
    )
    bands = tuple(
        RankBand((0.1, 0.4), (0.6, 0.9), 0.1, "finite-oracle")
        if len(a.values) == 2
        else RankBand((0.05, 0.2, 0.4), (0.5, 0.8, 0.95), 0.1, "finite-oracle")
        for a in raw.arms
    )
    region, additive, _ = project_bands(raw, "C", "T", bands, 2)
    feasible = 0
    distributions = [
        (Fraction(a, 4), Fraction(b, 4), Fraction(4 - a - b, 4))
        for a in range(5)
        for b in range(5 - a)
    ]
    for populations in product(distributions, repeat=3):
        valid = True
        for arm, band, probabilities in zip(raw.arms, bands, populations, strict=True):
            for x in (0, 1):
                k = sum(y <= x for y in arm.values)
                lo = 0 if k == 0 else band.lower[k - 1]
                hi = 1 if k == len(arm.values) else band.upper[k]
                if not lo <= sum(probabilities[: x + 1]) <= hi:
                    valid = False
        if not valid:
            continue
        weights = [Fraction(len(a.values), 7) for a in raw.arms]
        c = next(
            x
            for x in (0, 1, 2)
            if sum(w * sum(p[: x + 1]) for w, p in zip(weights, populations, strict=True))
            >= Fraction(raw.quantile)
        )
        means = {
            a.group_id: sum(min(x, c) * p for x, p in enumerate(probs))
            for a, probs in zip(raw.arms, populations, strict=True)
        }
        assert _contains(additive, means["T"] - means["C"])
        if means["C"] > 0:
            assert _contains(region, means["T"] / means["C"] - 1)
        feasible += 1
    assert feasible > 0
    if quantile == 0.9:
        assert region.lower.value is not None and region.upper.value is not None
    assert additive.lower.value is not None and additive.upper.value is not None
    assert additive.lower.value < additive.upper.value


@pytest.mark.parametrize("offset", [1.0, 1e16, 1e308])
def test_joint_cap_and_gap_limit_rational_oracle(offset):
    from increment.estimation._rank_bands import RankBand
    from increment.estimation.winsor import project_bands
    from increment.winsor import WinsorRawState

    step = math.ulp(offset) if offset > 1 else 1.0
    high = offset + step
    payload = _raw((offset, high), (offset, high), upper=high, quantile=0.3).model_dump()
    payload["support"]["lower"] = offset
    raw = WinsorRawState.model_validate(payload)
    band = RankBand((0.1, 0.4), (0.6, 0.9), 0.1, "rational-oracle")
    relative, additive, _ = project_bands(raw, "C", "T", (band, band), high)
    # The upper gap limit has p*=2q-a1; independent boxing instead uses b2.
    lower_mean = Fraction(offset) + Fraction(step) * (1 - (2 * Fraction(0.3) - Fraction(0.1)))
    upper_mean = Fraction(offset) + Fraction(step) * (1 - Fraction(0.1))
    exact_lo, exact_hi = lower_mean / upper_mean - 1, upper_mean / lower_mean - 1
    assert relative.lower.value is not None and relative.upper.value is not None
    assert Fraction(relative.lower.value) <= exact_lo
    assert Fraction(relative.upper.value) >= exact_hi
    assert abs(relative.lower.value - float(exact_lo)) <= math.ulp(float(exact_lo))
    assert abs(relative.upper.value - float(exact_hi)) <= math.ulp(float(exact_hi))
    assert _contains(additive, upper_mean - lower_mean)


def test_raw_frame_snapshot_support_and_wire():
    import narwhals as nw
    import pandas as pd

    from increment.estimation.winsor import raw_state_from_source
    from increment.frame import from_unit_summary
    from increment.winsor import WinsorRawState

    frame = pd.DataFrame(
        {
            "unit": list(range(8)),
            "group": ["C"] * 4 + ["T"] * 4,
            "revenue": [1.0, 2.0, 3.0, 9.0, 1.0, 2.0, 3.0, 5.0],
        }
    )
    source = from_unit_summary(
        frame,
        unit="unit",
        group="group",
        control="C",
        metrics=[
            {
                "name": "revenue",
                "winsorization": {
                    "upper_percentile": 0.75,
                    "support": {"lower": 0, "provenance": "External fixture support"},
                },
            }
        ],
    )
    metric = source.context.metrics[0]
    raw = raw_state_from_source(source, metric)
    assert raw.arm("C").values == (1, 2, 3, 9)
    assert raw.counts == (("C", 4), ("T", 4))
    frame.loc[3, "revenue"] = 999
    frame.loc[0, "revenue"] = 999
    assert raw_state_from_source(source, metric) == raw
    transformed = nw.from_native(source.unit_frame(metric), eager_only=True)
    assert transformed["y"].to_list() == [1, 2, 3, 3.5, 1, 2, 3, 3.5]
    assert WinsorRawState.model_validate_json(raw.model_dump_json()) == raw

    untrusted = {"native": frame}
    with pytest.raises(TypeError):
        raw_state_from_source(source, metric, **untrusted)

    from contextlib import contextmanager

    from increment.readouts import run
    from increment.sources import MomentSource

    captures = 0
    active = False

    class SnapshotWrapper:
        def __getattr__(self, name):
            return getattr(source, name)

        def unit_frame(self, *args, **kwargs):
            assert active, "raw outcomes loaded outside the snapshot lifetime"
            return source.unit_frame(*args, **kwargs)

        @contextmanager
        def readout_snapshot(self, **kwargs):
            nonlocal captures, active
            captures += 1
            assert captures == 1, "readout recursively requested another snapshot"
            active = True
            try:
                yield SnapshotWrapper()
            finally:
                active = False

    (readout,) = run(cast(MomentSource, SnapshotWrapper()))
    assert readout.require_lift().value == pytest.approx(0.0, abs=1e-14)

    # Deliberately pass an immutable value to the runtime mutation operation.
    def attempt_mutation(set_item: Callable[..., None]) -> None:
        set_item(raw.arms[0].values, 0, 3)

    with pytest.raises(TypeError):
        attempt_mutation(operator.setitem)


@pytest.mark.slow
def test_raw_warehouse_transform_roundtrip():
    import ibis
    import pyarrow as pa

    from increment.query.builders import winsorize_unit_totals
    from increment.semantics.models import MeanMetric, Winsorization

    con = ibis.duckdb.connect()
    try:
        table = con.create_table(
            "raw", obj=pa.table({"y": [1.0, 2.0, 3.0, 9.0, 1.0, 2.0, 3.0, 5.0]})
        )
        metric = MeanMetric(
            name="revenue",
            entity="unit",
            fact="revenue",
            winsorization=Winsorization(upper_percentile=0.75),
        )
        transformed = con.to_pyarrow(winsorize_unit_totals(table, metric))
        assert transformed["y_raw"].to_pylist() == [1, 2, 3, 9, 1, 2, 3, 5]
        assert transformed["y"].to_pylist() == [1, 2, 3, 3.5, 1, 2, 3, 3.5]
    finally:
        con.disconnect()


def test_confidence_set_wire_unbounded_undefined_and_no_posterior():
    from increment.estimation.results import LiftEstimate
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.tables import estimates_to_readout

    estimate = estimate_winsor_lift(_raw(quantile=0.99), "C", "T")
    restored = LiftEstimate.model_validate_json(estimate.model_dump_json())
    assert restored == estimate
    assert restored.confidence_set is not None
    assert restored.confidence_set.relative.upper.status == "unbounded"
    assert restored.confidence_set.relative.upper.value is None
    (table_row,) = estimates_to_readout([restored])
    assert table_row["higher"] is None
    assert table_row["higher_status"] == "unbounded"
    assert table_row["higher_reason"]
    from increment.winsor import RankReference

    assert isinstance(restored.confidence_set.reference, RankReference)
    assert (
        restored.confidence_set.reference.band_alpha
        + restored.confidence_set.reference.cutoff_alpha
        <= 0.05
    )
    # A confidence set is sampling evidence, not a posterior; see
    # docs/guides/priors-and-decisions.md:142-145.
    assert restored.chance_to_beat() is None
    reinverted = restored.reintervalize(0.01)
    assert reinverted.reference_kind == "confidence_set"
    assert reinverted.confidence_set is not None
    assert estimate.confidence_set is not None
    assert reinverted.confidence_set.raw == estimate.confidence_set.raw
    zero = estimate_winsor_lift(_raw((0, 0), (0, 0), upper=0), "C", "T")
    assert zero.lift is None
    assert zero.confidence_set is not None
    assert zero.confidence_set.relative.upper.status == "undefined"
    assert LiftEstimate.model_validate_json(zero.model_dump_json()) == zero
    assert estimates_to_readout([zero])[0]["higher_status"] == "undefined"


@pytest.mark.parametrize("field", ["relative_confidence_set", "relative_unavailable_reason"])
def test_rank_wire_rejects_competing_joint_relative_evidence(field):
    from increment.errors import CodedError
    from increment.estimation.results import (
        JointContrastReference,
        LiftEstimate,
        RelativeConfidenceSet,
    )
    from increment.estimation.winsor import estimate_winsor_lift

    estimate = estimate_winsor_lift(_raw(quantile=0.99), "C", "T")
    payload = estimate.model_dump(mode="json")
    payload[field] = (
        RelativeConfidenceSet(
            reference=JointContrastReference(a=0.2, c=1.0, var_a=0.01, var_c=0.01, cov_ac=0.0),
            alpha=0.05,
        ).model_dump(mode="json")
        if field == "relative_confidence_set"
        else "joint_covariance_indefinite"
    )
    with pytest.raises(CodedError) as error:
        LiftEstimate.model_validate(payload)
    assert error.value.code == "estimation.winsor.invalid_state"


def test_support_violation_and_quantile_bound_distinction():
    from increment.errors import CodedError
    from increment.winsor import WinsorRawState

    raw = _raw()
    payload = raw.model_dump()
    payload["support"]["upper"] = 5
    with pytest.raises(CodedError) as error:
        WinsorRawState.model_validate(payload)
    assert error.value.code == "estimation.winsor.support_violation"
    payload["support"]["upper"] = None
    payload["support"]["quantile_upper"] = 5
    assert WinsorRawState.model_validate(payload).arm("C").values[-1] == 9
    payload["support"]["lower"] = 2
    with pytest.raises(CodedError) as lower_error:
        WinsorRawState.model_validate(payload)
    assert lower_error.value.code == "estimation.winsor.support_violation"


def test_legacy_moments_refuse_before_iteration():
    from increment.errors import CodedError
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric, Winsorization

    def forbidden_rows():
        raise AssertionError("legacy summary must not be read")
        yield

    metric = MeanMetric(
        name="revenue",
        entity="unit",
        fact="revenue",
        winsorization=Winsorization(upper_percentile=0.99),
    )
    with pytest.raises(CodedError) as error:
        estimate_lift([metric], forbidden_rows(), "C")
    assert error.value.code == "estimation.winsor.raw_state_required"


def test_engine_percentile_defaults_match_explicit_fixed_policy():
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric, Winsorization

    state = _raw()
    metric = MeanMetric(
        name="revenue",
        entity="unit",
        fact="revenue",
        winsorization=Winsorization(
            upper_percentile=state.quantile,
            support=state.support,
            inference=state.inference,
        ),
    )
    raw = {"revenue": state}

    omitted = estimate_lift([metric], [], "C", raw_outcomes=raw)
    explicit = estimate_lift(
        [metric],
        [],
        "C",
        alpha=0.05,
        alternative="two-sided",
        null_lift=0.0,
        raw_outcomes=raw,
    )

    assert omitted == explicit
    (row,) = omitted.results
    assert row.require_lift().value == 0.0


def _ordinary_rows(*, study_id="e"):
    return [
        {
            "experiment_id": study_id,
            "metric": "ordinary",
            "group_id": group,
            "n": 4,
            "ref_y": mean,
            "cy1": 0.0,
            "cy2": 5.0,
        }
        for group, mean in (("C", 2.0), ("T", 3.0))
    ]


@pytest.mark.parametrize("population", ["assigned", "triggered"])
@pytest.mark.parametrize("native_frame", [False, True])
def test_mixed_winsor_identity_composes_with_explicit_population(population, native_frame):
    import pyarrow as pa

    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric, Winsorization

    raw_state = _raw(population=population)
    percentile = MeanMetric(
        name="revenue",
        entity="unit",
        fact="revenue",
        winsorization=Winsorization(
            upper_percentile=raw_state.quantile,
            support=raw_state.support,
            inference=raw_state.inference,
        ),
    )
    ordinary = MeanMetric(name="ordinary", entity="unit", fact="ordinary")

    rows = _ordinary_rows()
    result = estimate_lift(
        [percentile, ordinary],
        pa.Table.from_pylist(rows) if native_frame else rows,
        "C",
        raw_outcomes={"revenue": raw_state},
        summary_population=population,
    )
    assert {row.metric: row.require_lift().value for row in result.results} == pytest.approx(
        {"revenue": 0.0, "ordinary": 0.5}, rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"summary_population": None},
        {"summary_population": "triggered"},
    ],
)
def test_mixed_winsor_identity_refuses_missing_or_mismatched_population(kwargs):
    from increment.errors import CodedError
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric, Winsorization

    raw_state = _raw(population="assigned")
    percentile = MeanMetric(
        name="revenue",
        entity="unit",
        fact="revenue",
        winsorization=Winsorization(
            upper_percentile=raw_state.quantile,
            support=raw_state.support,
            inference=raw_state.inference,
        ),
    )
    ordinary = MeanMetric(name="ordinary", entity="unit", fact="ordinary")
    with pytest.raises(CodedError) as error:
        estimate_lift(
            [percentile, ordinary],
            _ordinary_rows(),
            "C",
            raw_outcomes={"revenue": raw_state},
            **kwargs,
        )
    assert error.value.code == "estimation.winsor.pool_mismatch"


@pytest.mark.parametrize("native_frame", [False, True])
def test_mixed_winsor_identity_refuses_study_mismatch_before_inference(native_frame):
    import pyarrow as pa

    from increment.errors import CodedError
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric, Winsorization

    raw_state = _raw()
    percentile = MeanMetric(
        name="revenue",
        entity="unit",
        fact="revenue",
        winsorization=Winsorization(
            upper_percentile=raw_state.quantile,
            support=raw_state.support,
            inference=raw_state.inference,
        ),
    )
    ordinary = MeanMetric(name="ordinary", entity="unit", fact="ordinary")

    rows = _ordinary_rows(study_id="other")
    with pytest.raises(CodedError) as error:
        estimate_lift(
            [percentile, ordinary],
            pa.Table.from_pylist(rows) if native_frame else rows,
            "C",
            raw_outcomes={"revenue": raw_state},
            summary_population="assigned",
        )
    assert error.value.code == "estimation.winsor.pool_mismatch"


def test_mixed_winsor_identity_retains_one_use_summary_iterable():
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric, Winsorization

    raw_state = _raw()
    percentile = MeanMetric(
        name="revenue",
        entity="unit",
        fact="revenue",
        winsorization=Winsorization(
            upper_percentile=raw_state.quantile,
            support=raw_state.support,
            inference=raw_state.inference,
        ),
    )
    ordinary = MeanMetric(name="ordinary", entity="unit", fact="ordinary")
    rows = (row for row in _ordinary_rows())
    result = estimate_lift(
        [percentile, ordinary],
        rows,
        "C",
        raw_outcomes={"revenue": raw_state},
        summary_population="assigned",
    )
    assert {row.metric: row.require_lift().value for row in result.results} == pytest.approx(
        {"revenue": 0.0, "ordinary": 0.5}, rel=1e-12, abs=1e-12
    )


def test_mixed_winsor_method_admission_precedes_summary_consumption():
    from increment.errors import CodedError
    from increment.estimation.engine import Method, estimate_lift
    from increment.semantics.models import MeanMetric, Winsorization

    raw_state = _raw()
    percentile = MeanMetric(
        name="revenue",
        entity="unit",
        fact="revenue",
        winsorization=Winsorization(
            upper_percentile=raw_state.quantile,
            support=raw_state.support,
            inference=raw_state.inference,
        ),
    )
    ordinary = MeanMetric(name="ordinary", entity="unit", fact="ordinary")

    def unavailable_rows():
        raise AssertionError("invalid methods consumed the summary")
        yield {}

    with pytest.raises(CodedError) as error:
        estimate_lift(
            [percentile, ordinary],
            unavailable_rows(),
            "C",
            methods=[Method(name="unadjusted"), Method(name="unadjusted")],
            raw_outcomes={"revenue": raw_state},
            summary_population="assigned",
        )
    assert error.value.code == "estimation.engine.method_names_unique"


@pytest.mark.parametrize("field,value", [("study_id", "other"), ("population", "triggered")])
def test_percentile_raw_states_must_share_identity(field, value):
    from increment.errors import CodedError
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric, Winsorization

    states = (_raw(), _raw().model_copy(update={"metric": "another", field: value}))
    metrics = [
        MeanMetric(
            name=state.metric,
            entity="unit",
            fact=state.metric,
            winsorization=Winsorization(
                upper_percentile=state.quantile,
                support=state.support,
                inference=state.inference,
            ),
        )
        for state in states
    ]
    with pytest.raises(CodedError) as error:
        estimate_lift(metrics, [], "C", raw_outcomes={state.metric: state for state in states})
    assert error.value.code == "estimation.winsor.pool_mismatch"


def test_moments_source_raw_refusal():
    from increment.errors import CodedError
    from increment.semantics.models import AnalysisPlan, MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(name="revenue", entity="unit", fact="revenue")
    source = MomentsSource([], metrics=[metric], study_id="e", plan=AnalysisPlan())
    with pytest.raises(CodedError) as error:
        source.unit_frame(metric, outcome_stage="raw")
    assert error.value.code == "estimation.winsor.raw_state_required"


@pytest.mark.parametrize(
    "method", ["positive-log-kernel-bootstrap-t-v1", "joint-rank-projection-v1"]
)
def test_experimental_winsor_methods_cannot_emit_decision_evidence(method):
    from increment.estimation.decision_types import ArmHypothesisKey
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric, Winsorization
    from increment.winsor import WinsorInferenceSpec

    raw = _raw(upper=10).model_copy(update={"inference": WinsorInferenceSpec(method=method)})
    metric = MeanMetric(
        name=raw.metric,
        entity="unit",
        fact="outcome",
        winsorization=Winsorization(
            upper_percentile=raw.quantile, support=raw.support, inference=raw.inference
        ),
    )
    computation = estimate_lift([metric], [], "C", raw_outcomes={raw.metric: raw})
    (result,) = computation.results
    assert result.confidence_set is not None
    assert result.confidence_set.raw.inference.method == method
    assert computation.evidence == {}
    failure = computation.failures[ArmHypothesisKey(raw.metric, "T", "itt")]
    assert failure.code == "evidence.experimental_reference"
    assert failure.context["status"] == "experimental"


@pytest.mark.parametrize("include_plain", [False, True])
def test_public_readout_uses_raw_pool(include_plain):
    import pyarrow as pa

    from increment import readouts
    from increment.frame import from_unit_summary

    table = pa.table(
        {
            "unit": list(range(8)),
            "group": ["C"] * 4 + ["T"] * 4,
            "revenue": [1.0, 2.0, 3.0, 9.0, 1.0, 2.0, 3.0, 5.0],
            "plain": [2.0, 3.0, 5.0, 7.0, 3.0, 5.0, 7.0, 11.0],
        }
    )
    source = from_unit_summary(
        table,
        unit="unit",
        group="group",
        control="C",
        metrics=[
            {
                "name": "revenue",
                "winsorization": {
                    "upper_percentile": 0.75,
                    "support": {"lower": 0, "upper": 10, "provenance": "External fixture support"},
                },
            }
        ]
        + ([{"name": "plain"}] if include_plain else []),
    )
    rows = readouts.run(source)
    assert [result.metric for result in rows] == ["revenue"] + (["plain"] if include_plain else [])
    row = rows[0]
    assert row.reference_kind == "confidence_set"
    assert row.confidence_set is not None
    assert row.confidence_set.raw.counts == (("C", 4), ("T", 4))
    assert row.abs_diff == 0
    if include_plain:
        plain_source = from_unit_summary(
            table, unit="unit", group="group", control="C", metrics={"plain": "mean"}
        )
        [plain] = readouts.run(plain_source)
        actual, expected = rows[1].require_lift(), plain.require_lift()
        assert (actual.value, actual.lb, actual.ub) == pytest.approx(
            (expected.value, expected.lb, expected.ub)
        )


@pytest.mark.slow
@pytest.mark.parametrize("volatile", [False, True])
@pytest.mark.parametrize(
    "inference_method", ["positive-log-kernel-bootstrap-t-v1", "joint-rank-projection-v1"]
)
def test_definitions_source_preserves_raw_pool_and_wire(tmp_path, volatile, inference_method):  # noqa: PLR0915
    from datetime import UTC, datetime

    import ibis
    import narwhals as nw
    import pyarrow as pa

    from increment import SourceSnapshotEvidence
    from increment.analysis import Analysis
    from increment.estimation.winsor import raw_state_from_source
    from increment.results import LiftEstimate
    from increment.winsor import WinsorRawState

    definitions = tmp_path / "definitions.yml"
    definitions.write_text("""
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {name: revenue, column: value}
      - {name: enrolled, column: null}
      - {name: activated, column: null}
exposures:
  - {name: assignment, fact: enrolled}
  - {name: activated, fact: activated}
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    aggregation: sum
    winsorization:
      upper_percentile: 0.75
      support: {lower: 0, provenance: External fixture support}
experiments:
  - name: e
    exposure: assignment
    trigger: activated
    unit: user_id
    start: 2024-01-01T00:00:00
    control_group: C
    plan: {primary: revenue}
""")
    definitions.write_text(
        definitions.read_text().replace(
            "upper_percentile: 0.75",
            f"upper_percentile: 0.75\n      inference: {{method: {inference_method}, seed: 803, stream: 17}}",
        )
    )
    rows: list[dict[str, str | datetime | float | None]] = [
        {
            "user_id": str(i),
            "experiment_id": "e",
            "group_id": "C" if i < 4 else "T",
            "event": "revenue",
            "ts": datetime(2024, 1, 2),
            "value": float(y),
        }
        for i, y in enumerate((1, 2, 3, 9, 1, 2, 3, 5))
    ]
    rows.extend(
        {**row, "event": "enrolled", "ts": datetime(2024, 1, 1), "value": None}
        for row in tuple(rows)
    )
    rows.extend(
        {**row, "event": "activated", "ts": datetime(2024, 1, 1), "value": None}
        for row in tuple(rows)
        if row["event"] == "enrolled" and row["user_id"] in {"0", "1", "4", "5"}
    )
    if volatile:
        definitions.write_text(
            definitions.read_text().replace(
                "sql: SELECT * FROM events", "sql: SELECT * FROM changing_events"
            )
        )
    con = ibis.duckdb.connect()
    try:
        con.create_table("events", obj=rows)
        if volatile:
            con.raw_sql("CREATE SEQUENCE revision START 1")
            con.raw_sql("""CREATE VIEW changing_events AS
                WITH state AS MATERIALIZED (SELECT nextval('revision') AS revision)
                SELECT events.* EXCLUDE (value, ts), value * revision AS value,
                       ts + CAST(revision - 1 AS INTEGER) * INTERVAL '1 day' AS ts
                FROM events CROSS JOIN state""")
        from increment.errors import IncrementWarning

        with pytest.warns(IncrementWarning):
            analysis = Analysis.from_definitions(
                "e",
                definitions,
                con,
                store="none",
                source_snapshot_evidence=SourceSnapshotEvidence(
                    observation_cutoff_ts=datetime(2031, 1, 1, tzinfo=UTC),
                    complete_through_by_feed={"events": datetime(2031, 1, 1, tzinfo=UTC)},
                ),
            )
        from tests.analysis_factory import _native_source

        source = _native_source(analysis)
        metric = source.context.metrics[0]
        raw = raw_state_from_source(source, metric)
        assert raw.inference.method == inference_method
        assert (raw.inference.seed, raw.inference.stream) == (803, 17)
        assert raw.arm("C").values == (1, 2, 3, 9)
        assert raw.arm("T").values == (1, 2, 3, 5)
        assert raw.allocation == (("C", 4, 8), ("T", 4, 8))
        assert raw.missingness == "measure-unit-inclusion-v1:sum:observable-windows"
        assert WinsorRawState.model_validate_json(raw.model_dump_json()) == raw
        triggered = raw_state_from_source(source.triggered_source(), metric)
        assert triggered.population == "triggered"
        assert triggered.counts == (("C", 2), ("T", 2))
        assert triggered.arm("C").values == ((2, 4) if volatile else (1, 2))
        if not volatile:
            native_results: dict[str, LiftEstimate] = {}
            for row in analysis.run():
                assert isinstance(row, LiftEstimate)
                native_results[row.analysis_population] = row
            assert set(native_results) == {"assigned", "triggered"}
            assigned_region = native_results["assigned"].confidence_set
            triggered_region = native_results["triggered"].confidence_set
            assert assigned_region is not None and triggered_region is not None
            assert assigned_region.raw == raw
            assert triggered_region.raw == triggered
        from increment import readouts
        from increment.query.artifact_publish import artifact_context
        from increment.query.artifact_reader import ArtifactMomentSource
        from increment.query.session import WarehouseArtifactStore
        from increment.query.source import open_artifact
        from increment.semantics.artifact import (
            AssignmentCountsRequest,
            TriggerMeasureStatsRequest,
            TriggerPopulationRequest,
        )
        from increment.semantics.loader import load

        with pytest.warns(IncrementWarning):
            loaded = load(definitions)
        context = artifact_context(loaded, loaded.experiments[0], "error")
        store = WarehouseArtifactStore(con, schema_name="artifacts")
        reference = analysis.publish_unit_day_artifact(
            store,
            extensions=[
                TriggerPopulationRequest(trigger_name="activated"),
                AssignmentCountsRequest(populations=("assigned", "triggered")),
                TriggerMeasureStatsRequest(trigger_name="activated", metric_names=(metric.name,)),
            ],
        )
        # Two raw reads and publication each capture the volatile stream once.
        revision = 3 if volatile else 1
        con.raw_sql("UPDATE events SET value = 999, ts = TIMESTAMP '2030-01-01'")
        with open_artifact(store, reference, expected_context=context) as adopted:
            trusted = adopted.context.metrics[0]
            restored = raw_state_from_source(adopted, trusted)
            assert restored.arm("C").values == tuple(revision * y for y in (1, 2, 3, 9))
            assert restored.arm("T").values == tuple(revision * y for y in (1, 2, 3, 5))
            assert restored.inference == raw.inference
            from increment.semantics.models import MeanMetric

            assert isinstance(trusted, MeanMetric) and trusted.winsorization is not None
            assert trusted.winsorization.inference == raw.inference
            assert restored.support == raw.support
            assert restored.allocation == raw.allocation
            assert restored.missingness == raw.missingness
            transformed = nw.from_native(adopted.unit_frame(trusted), eager_only=True)
            assert max(transformed["y"].to_list()) == revision * 3.5
            expected_day = datetime(2024, 1, 1 + revision).date()
            assert adopted.manifest.last_ds == expected_day
            assert adopted.manifest.measures[0].freshness.loaded_through == expected_day
            assert adopted.manifest.measures[0].event_horizon == expected_day
            [result] = readouts.run(adopted)
            assert result.confidence_set is not None
            assert result.confidence_set.raw == restored
            assert result.confidence_set.method == inference_method
            assert LiftEstimate.model_validate_json(result.model_dump_json()) == result
            if not volatile:
                assert result.confidence_set == native_results["assigned"].confidence_set
                assert (
                    result.reintervalize(0.1).confidence_set
                    == native_results["assigned"].reintervalize(0.1).confidence_set
                )
            assert result.abs_diff == 0
            assert WinsorRawState.model_validate_json(restored.model_dump_json()) == restored
            restored_triggered = raw_state_from_source(adopted.triggered_source(), trusted)
            assert restored_triggered.population == "triggered"
            assert restored_triggered.counts == triggered.counts
            assert restored_triggered.arm("C").values == (revision, 2 * revision)
            assert adopted.triggered_counts()[1] == {"C": 2, "T": 2}
        with ArtifactMomentSource.open(store, reference, expected_context=context) as reader:
            trusted = reader.context.metrics[0]
            native = reader.unit_frame(trusted, outcome_stage="raw")
            assert isinstance(native, pa.Table)
            frame = nw.from_native(native, eager_only=True)
            assert sorted(frame["y"].to_list()) == sorted(
                revision * y for y in (1, 2, 3, 9, 1, 2, 3, 5)
            )

    finally:
        con.disconnect()


@pytest.mark.parametrize("missing,expected", [("zero", (0, 1)), ("drop", (1,))])
def test_raw_missingness_precedes_support_validation(missing, expected):
    import pyarrow as pa

    from increment.errors import IncrementWarning
    from increment.estimation.winsor import raw_state_from_source
    from increment.frame import from_unit_summary
    from tests.warning_codes import warning_codes

    def _build():
        return from_unit_summary(
            pa.table(
                {"u": [0, 1, 2, 3], "g": ["C", "C", "T", "T"], "revenue": [None, 1.0, 2.0, 4.0]}
            ),
            unit="u",
            group="g",
            control="C",
            metrics=[
                {
                    "name": "revenue",
                    "missing": missing,
                    "winsorization": {
                        "upper_percentile": 0.75,
                        "support": {"lower": 0, "provenance": "External fixture support"},
                    },
                }
            ],
        )

    if missing == "drop":
        with pytest.warns(IncrementWarning) as rec:
            source = _build()
        assert "frame.validation.metric_missing_drop" in warning_codes(rec)
    else:
        source = _build()
    raw = raw_state_from_source(source, source.context.metrics[0])
    assert raw.arm("C").values == expected
    assert raw.missingness == missing


def test_finite_reference_roundtrip_and_allocation_mutation():
    from increment.errors import CodedError
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.results import LiftEstimate, WinsorConfidenceSet
    from increment.winsor import WinsorRawState

    raw = _raw((1, 1), (2, 2), upper=2)
    bounded = raw.model_dump()
    bounded["support"]["lower"] = 1
    raw = WinsorRawState.model_validate(bounded)
    row = estimate_winsor_lift(raw, "C", "T")
    assert isinstance(row.confidence_set, WinsorConfidenceSet)
    assert row.confidence_set.relative.lower.status == "finite"
    assert row.confidence_set.relative.upper.status == "finite"
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row
    payload = raw.model_dump()
    payload["allocation"] = (("C", 2, 5), ("T", 2, 5))
    with pytest.raises(CodedError) as error:
        WinsorRawState.model_validate(payload)
    assert error.value.code == "estimation.winsor.pool_mismatch"


def test_evaluation_reducer_keeps_nonfinite_rank_outcomes_in_denominator():
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.simulate.runner import _lift_outcome, _reduce_key
    from increment.winsor import WinsorRawState

    payload = _raw((1, 2), (1, 2), upper=2).model_dump()
    payload["support"]["lower"] = 1
    states = (
        WinsorRawState.model_validate(payload),
        _raw(quantile=0.99),
        _raw((0, 0), (0, 0), upper=0),
    )
    outcomes = []
    for state in states:
        row = estimate_winsor_lift(state, "C", "T")
        outcomes.append(_lift_outcome(row.lift, confidence_set=row.confidence_set))
    result = _reduce_key(outcomes, truth=0)
    assert result.attempted == 3
    assert result.point_estimable == 2
    assert result.interval_estimable == 2
    assert result.excluded == 1
    assert result.coverage_unconditional == pytest.approx(2 / 3)
    assert result.exclusion_reasons == {"control_mean_identically_zero": 1}
    assert result.interval_unavailable_reasons == {}


@pytest.mark.parametrize("alpha", [math.ulp(0.0), 1e-300, 0.05])
def test_exact_error_budget_and_wire_at_extreme_alpha(alpha):
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.results import LiftEstimate
    from increment.tables import estimates_to_readout
    from increment.winsor import WinsorRawState

    payload = _raw((1, 1), (2, 2), upper=2).model_dump()
    payload["support"]["lower"] = 1
    row = estimate_winsor_lift(WinsorRawState.model_validate(payload), "C", "T", alpha=alpha)
    region = row.confidence_set
    assert region is not None
    from increment.winsor import RankReference

    assert isinstance(region.reference, RankReference)
    assert Fraction(region.reference.band_alpha) + Fraction(
        region.reference.cutoff_alpha
    ) <= Fraction(alpha)
    assert region.alpha == alpha
    assert region.relative.lower.status == region.relative.upper.status == "finite"
    assert estimates_to_readout([row])[0]["higher"] == region.upper
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row


@pytest.mark.parametrize(
    "field",
    [
        "relative",
        "additive",
        "cutoff",
        "point",
        "additive_point",
        "alpha",
        "band_identity",
        "band_alpha",
        "cutoff_alpha",
    ],
)
def test_rank_region_wire_recomputes_all_observable_fields(field):
    from increment.errors import CodedError
    from increment.estimation.winsor import joint_confidence_set
    from increment.winsor import WinsorConfidenceSet, WinsorRawState

    state = _raw((1, 2, 3), (2, 3, 4), upper=4).model_dump()
    state["support"]["lower"] = 1
    region = joint_confidence_set(WinsorRawState.model_validate(state), "C", "T")
    assert WinsorConfidenceSet.model_validate_json(region.model_dump_json()) == region
    payload = region.model_dump(mode="json")
    if field in ("relative", "additive", "cutoff"):
        payload[field]["upper"]["value"] += 1
    elif field == "band_identity":
        payload["reference"][field][0] = "forged"
    elif field in ("band_alpha", "cutoff_alpha"):
        payload["reference"][field] /= 2
    else:
        payload[field] += 0.01
    with pytest.raises(CodedError) as error:
        WinsorConfidenceSet.model_validate(payload)
    assert error.value.code == "estimation.winsor.invalid_state"


@pytest.mark.slow
def test_rank_supported_size_boundary_and_oversized_refusal():
    from decimal import Decimal

    from increment import _winsor_rank as _rank_bands
    from increment.errors import CapabilityError
    from increment.estimation.winsor import joint_confidence_set

    n = _rank_bands.MAX_RANK_ARM_SIZE
    band = _rank_bands.simultaneous_rank_band(n, 0.025)
    lower, _ = _rank_bands.crossing_probability(band.lower, band.upper)
    assert lower >= 1 - Decimal.from_float(0.025)

    with pytest.raises(CapabilityError) as error:
        _rank_bands.simultaneous_rank_band(n + 1, 0.025)
    assert error.value.code == "estimation.winsor.rank_size_unsupported"
    with pytest.raises(CapabilityError) as error:
        joint_confidence_set(_raw(tuple(range(n + 1)), (1, 2)), "C", "T")
    assert error.value.code == "estimation.winsor.rank_size_unsupported"


@pytest.mark.parametrize(
    "lower,upper,point,available,covered",
    [
        ("finite", "finite", 0.0, True, True),
        ("unbounded", "unbounded", 0.0, True, True),
        ("finite", "unbounded", 0.0, True, True),
        ("unbounded", "finite", None, True, True),
        ("finite", "unbounded", None, True, True),
        ("undefined", "finite", 0.0, False, False),
        ("finite", "undefined", None, False, False),
    ],
)
def test_reducer_distinguishes_endpoint_statuses(lower, upper, point, available, covered):
    from increment.simulate.runner import _lift_outcome, _reduce_key
    from increment.winsor import SetEndpoint, SetInterval, WinsorConfidenceSet

    # Isolate the reducer's status adapter from statistical construction.
    region = SetInterval(
        lower=SetEndpoint(
            status=lower,
            value=-1 if lower == "finite" else None,
            reason=None if lower == "finite" else "test_lower",
        ),
        upper=SetEndpoint(
            status=upper,
            value=1 if upper == "finite" else None,
            reason=None if upper == "finite" else "test_upper",
        ),
    )
    confidence_set = WinsorConfidenceSet.model_construct(
        relative=region,
        additive=region,
        point=point,
        additive_point=point,
    )
    for scale in ("relative", "additive"):
        outcome = _lift_outcome(None, confidence_set=confidence_set, winsor_scale=scale)
        assert (outcome.lower_status, outcome.upper_status) == (lower, upper)
        result = _reduce_key([outcome], truth=0)
        assert result.confidence_set_estimable == int(available)
        assert result.set_coverage_unconditional == float(covered)
        assert result.excluded == int(point is None and not available)
        assert result.interval_estimable == int(point is not None and available)
        if not available:
            reasons = (
                result.exclusion_reasons if point is None else result.interval_unavailable_reasons
            )
            assert reasons == {"test_lower" if lower == "undefined" else "test_upper": 1}


@pytest.mark.parametrize("field", ["relative", "additive", "cutoff"])
@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("mutation", ["value", "status", "reason"])
def test_rank_wire_rejects_each_endpoint_mutation(field, side, mutation):
    from increment.errors import CodedError
    from increment.estimation.winsor import joint_confidence_set
    from increment.winsor import WinsorConfidenceSet, WinsorRawState

    state = _raw((1, 2, 3), (2, 3, 4), upper=4).model_dump()
    state["support"]["lower"] = 1
    region = joint_confidence_set(WinsorRawState.model_validate(state), "C", "T")
    payload = region.model_dump(mode="json")
    endpoint = payload[field][side]
    if mutation == "value":
        endpoint["value"] += -1 if side == "lower" else 1
    elif mutation == "status":
        endpoint.update(status="unbounded", value=None, reason="forged")
    else:
        endpoint["reason"] = "forged"
    with pytest.raises(CodedError) as error:
        WinsorConfidenceSet.model_validate(payload)
    assert error.value.code == "estimation.winsor.invalid_state"
