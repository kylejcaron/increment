"""Independent finite arithmetic and portable full-procedure references."""

import math

import numpy as np
import pytest


def _state(*, stream=0):
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState

    return WinsorRawState(
        metric="revenue",
        study_id="e",
        missingness="error",
        quantile=0.75,
        inference=WinsorInferenceSpec(stream=stream),
        arms=(
            RawArm(group_id="C", values=(1, 2, 3, 9)),
            RawArm(group_id="T", values=(2, 3, 5, 8, 12)),
            RawArm(group_id="O", values=(1, 4, 7)),
        ),
    )


def test_kernel_moments_against_normal_integrals():
    from scipy.integrate import quad
    from scipy.stats import norm

    from increment.estimation._winsor_bootstrap import pilot_parts
    from increment.winsor import PositiveLogPilot

    pilot = PositiveLogPilot(group_id="C", log_centers=(0, math.log(4)), bandwidth=0.7)
    c = 3.0
    cdf, density, first, second = pilot_parts(pilot, c)
    expected = []
    for k in (1, 2):
        pieces = []
        for z in pilot.log_centers:
            boundary = (math.log(c) - z) / pilot.bandwidth
            lower = quad(
                lambda u, k=k, z=z: math.exp(k * (z + pilot.bandwidth * u)) * norm.pdf(u),
                -12,
                boundary,
                epsabs=1e-12,
            )[0]
            upper = c**k * norm.sf(boundary)
            pieces.append(lower + upper)
        expected.append(sum(pieces) / 2)
    assert (first, second) == pytest.approx(expected, rel=1e-11)
    assert cdf == pytest.approx(
        sum(norm.cdf((math.log(c) - z) / 0.7) for z in pilot.log_centers) / 2
    )
    step = 1e-5
    numerical_density = (pilot_parts(pilot, c + step)[0] - pilot_parts(pilot, c - step)[0]) / (
        2 * step
    )
    assert density == pytest.approx(numerical_density, rel=1e-8)


def test_pilot_population_target_uses_allocation_and_not_type7():
    from increment.estimation._winsor_bootstrap import (
        fit_positive_log_pilot,
        pilot_parts,
        pilot_population_target,
    )
    from increment.estimation.winsor import _linear_cutoff

    raw = _state()
    pilots = fit_positive_log_pilot(raw)
    c, ell, delta = pilot_population_target(pilots, raw.quantile, "C", "T")
    parts = {p.group_id: pilot_parts(p, c) for p in pilots}
    assert sum(w * parts[g][0] for g, w in raw.weights) == pytest.approx(raw.quantile, abs=1e-12)
    assert ell == pytest.approx(math.log(parts["T"][2] / parts["C"][2]))
    assert delta == pytest.approx(parts["T"][2] - parts["C"][2])
    assert c != pytest.approx(_linear_cutoff(raw))


@pytest.mark.parametrize("counts", [(50, 200, 80), (200, 50, 80)])
@pytest.mark.parametrize("relative", [False, True])
def test_population_contamination_derivative_each_pool_arm(counts, relative):
    from scipy.optimize import brentq

    bounds = (1.0, 2.0, 3.0)
    weights = np.asarray(counts) / sum(counts)
    q = 0.7

    def target(arm, epsilon, y):
        def cdf(c, j):
            original = min(c / bounds[j], 1)
            return original if j != arm else (1 - epsilon) * original + epsilon * (y <= c)

        c = brentq(lambda x: sum(weights[j] * cdf(x, j) for j in range(3)) - q, 0, 3)
        means = []
        for j, high in enumerate(bounds):
            m = high / 2 if c >= high else c - c * c / (2 * high)
            means.append(m if j != arm else (1 - epsilon) * m + epsilon * min(y, c))
        return c, means, math.log(means[1] / means[0]) if relative else means[1] - means[0]

    c, means, theta = target(0, 0, 0)
    cdfs = [min(c / high, 1) for high in bounds]
    a = [-1 / means[0], 1 / means[1], 0] if relative else [-1, 1, 0]
    derivative = sum(aj * (1 - f) for aj, f in zip(a, cdfs, strict=True))
    density = sum(w / high for w, high in zip(weights, bounds, strict=True) if c < high)
    for g in range(3):
        for y in (0.2, 2.8):
            expected = a[g] * (min(y, c) - means[g]) + derivative * weights[g] / density * (
                cdfs[g] - (y <= c)
            )
            observed = (target(g, 1e-7, y)[2] - theta) / 1e-7
            assert observed == pytest.approx(expected, rel=2e-5, abs=2e-6)


def _independent_scores(samples, q, ci, ti):
    """Literal scalar score algebra, independently of vectorized production."""
    from scipy.stats import norm

    cutoff = float(np.quantile(np.concatenate(samples), q, method="linear"))
    means = [float(np.minimum(y, cutoff).mean()) for y in samples]
    cdfs = [float((y <= cutoff).mean()) for y in samples]
    total = sum(len(y) for y in samples)
    density = 0.0
    for y in samples:
        z = np.log(y)
        h = 1.06 * z.std(ddof=1) * len(y) ** -0.2
        density += float(norm.pdf((math.log(cutoff) - z) / h).sum()) / (cutoff * h * total)
    ses = []
    for relative in (True, False):
        a = [0.0] * len(samples)
        a[ci], a[ti] = (-1 / means[ci], 1 / means[ti]) if relative else (-1, 1)
        A = sum(v * (1 - f) for v, f in zip(a, cdfs, strict=True))
        variance = 0.0
        for g, y in enumerate(samples):
            psi = a[g] * (np.minimum(y, cutoff) - means[g]) + A * len(y) / total / density * (
                cdfs[g] - (y <= cutoff)
            )
            variance += float(np.var(psi, ddof=1)) / len(y)
        ses.append(math.sqrt(variance))
    return cutoff, math.log(means[ti] / means[ci]), means[ti] - means[ci], *ses


def test_vectorized_refit_matches_explicit_scores_and_recomputes_cutoff():
    from increment.estimation._winsor_bootstrap import full_procedure_statistics

    samples = (
        np.asarray([[1.0, 2.0, 4.0], [1.0, 8.0, 12.0]]),
        np.asarray([[2.0, 3.0, 8.0, 10.0], [2.0, 3.0, 4.0, 5.0]]),
        np.asarray([[3.0, 9.0], [1.0, 6.0]]),
    )
    actual = full_procedure_statistics(samples, 0.75, 0, 1)
    assert actual.cutoff[0] != actual.cutoff[1]
    for b in range(2):
        expected = _independent_scores(tuple(y[b] for y in samples), 0.75, 0, 1)
        assert (
            actual.cutoff[b],
            actual.log_relative[b],
            actual.additive[b],
            actual.log_se[b],
            actual.additive_se[b],
        ) == pytest.approx(expected, rel=1e-12)


@pytest.mark.slow
def test_full_reference_stream_centering_wire_and_reinversion():
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.results import LiftEstimate
    from increment.winsor import BootstrapReference

    raw = _state(stream=23)
    row = estimate_winsor_lift(raw, "C", "T")
    confidence_set = row.confidence_set
    assert confidence_set is not None
    reference = confidence_set.reference
    assert isinstance(reference, BootstrapReference)
    assert reference.failure_indices == ()
    samples = []
    for g, p in enumerate(reference.pilots):
        centers = np.random.Generator(
            np.random.PCG64DXSM(np.random.SeedSequence(1729, spawn_key=(23, g, 0)))
        )
        noise = np.random.Generator(
            np.random.PCG64DXSM(np.random.SeedSequence(1729, spawn_key=(23, g, 1)))
        )
        indices = centers.integers(len(p.log_centers), size=(64, len(p.log_centers)))
        draws = np.exp(
            np.asarray(p.log_centers)[indices] + p.bandwidth * noise.standard_normal(indices.shape)
        )
        samples.append(draws[0])
    ci = [a.group_id for a in raw.arms].index("C")
    ti = [a.group_id for a in raw.arms].index("T")
    _, ell, delta, sl, sa = _independent_scores(tuple(samples), raw.quantile, ci, ti)
    assert reference.log_relative.roots[0] == pytest.approx(
        (ell - reference.log_relative.pilot_target) / sl
    )
    assert reference.additive.roots[0] == pytest.approx(
        (delta - reference.additive.pilot_target) / sa
    )

    def _bootstrap_reference(state):
        result = estimate_winsor_lift(state, "C", "T")
        candidate = result.confidence_set
        assert candidate is not None
        resolved = candidate.reference
        assert isinstance(resolved, BootstrapReference)
        return resolved

    assert reference == _bootstrap_reference(raw)
    other = _bootstrap_reference(_state(stream=24))
    assert reference.log_relative.roots != other.log_relative.roots
    for series, interval, relative in (
        (reference.log_relative, confidence_set.relative, True),
        (reference.additive, confidence_set.additive, False),
    ):
        roots = sorted(x for x in series.roots if x is not None)
        lo, hi = series.point - roots[-50] * series.se, series.point - roots[49] * series.se
        expected = (math.expm1(lo), math.expm1(hi)) if relative else (lo, hi)
        assert (interval.lower.value, interval.upper.value) == pytest.approx(expected)
    restored = LiftEstimate.model_validate_json(row.model_dump_json())
    wider = restored.reintervalize(0.01)
    assert restored.confidence_set is not None and wider.confidence_set is not None
    assert isinstance(wider.confidence_set.reference, BootstrapReference)
    assert wider.confidence_set.reference == reference
    assert wider.confidence_set.lower is not None and restored.confidence_set.lower is not None
    assert wider.confidence_set.upper is not None and restored.confidence_set.upper is not None
    assert wider.confidence_set.lower <= restored.confidence_set.lower
    assert wider.confidence_set.upper >= restored.confidence_set.upper
    unresolved = restored.reintervalize(1e-300)
    assert unresolved.confidence_set is not None
    assert unresolved.confidence_set.relative.upper.reason == "bootstrap_tail_unresolved"
    assert restored.p_value() == row.p_value()


@pytest.mark.slow
def test_failed_root_is_persisted_without_retry_and_invalid_wire_refuses():
    from increment.errors import CodedError
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.winsor import BootstrapReference, WinsorConfidenceSet

    raw = _state()
    base = estimate_winsor_lift(raw, "C", "T").confidence_set
    assert base is not None
    ref = base.reference
    assert isinstance(ref, BootstrapReference)
    payload = ref.model_dump()
    roots = list(payload["log_relative"]["roots"])
    roots[17] = None
    payload["log_relative"]["roots"] = roots
    payload["failure_indices"] = (17,)
    failed = BootstrapReference.model_validate(payload)
    region = estimate_winsor_lift(raw, "C", "T", reference=failed).confidence_set
    assert region is not None
    assert (
        region.relative.lower.reason
        == region.additive.upper.reason
        == "bootstrap_replicate_failure"
    )
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric, Winsorization

    metric = MeanMetric(
        name="revenue",
        entity="unit",
        fact="revenue",
        winsorization=Winsorization(upper_percentile=raw.quantile, inference=raw.inference),
    )
    result = estimate_lift(
        [metric],
        [],
        "C",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {"revenue": raw},
        {("revenue", "T"): failed},
    )
    retained = next(row for row in result.results if row.group_id == "T")
    assert retained.confidence_set is not None
    assert retained.confidence_set.relative.lower.reason == "bootstrap_replicate_failure"
    assert region.relative.lower.status == "undefined"
    bad = base.model_dump()
    bad["relative"]["lower"]["value"] = -0.999
    with pytest.raises(CodedError) as error:
        WinsorConfidenceSet.model_validate(bad)
    assert error.value.code == "estimation.winsor.invalid_state"


@pytest.mark.parametrize(
    "values,reason",
    [
        ((0, 1), "pilot_nonpositive_outcome"),
        ((-1, 1), "pilot_nonpositive_outcome"),
        ((2, 2), "pilot_degenerate"),
    ],
)
def test_pilot_applicability_is_explicit(values, reason):
    from increment.errors import CodedError
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.winsor import WinsorRawState

    payload = _state().model_dump()
    payload["arms"][0]["values"] = values
    payload.pop("allocation")
    with pytest.raises(CodedError) as error:
        estimate_winsor_lift(WinsorRawState.model_validate(payload), "C", "T")
    assert error.value.code == "estimation.winsor." + reason


def test_unresolved_neighboring_logs_refuse_without_density_floor():
    from increment.errors import CodedError
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.winsor import WinsorRawState

    payload = _state().model_dump()
    payload["arms"][0]["values"] = (1e308, math.nextafter(1e308, math.inf))
    payload.pop("allocation")
    with pytest.raises(CodedError) as error:
        estimate_winsor_lift(WinsorRawState.model_validate(payload), "C", "T")
    assert error.value.code == "estimation.winsor.pilot_degenerate"


def test_exact_fixed_count_permutation_all_70_and_ties():
    from increment.estimation._winsor_permutation import conditional_permutation_test
    from increment.winsor import RawArm, WinsorPermutationTest, WinsorRawState
    from tests._i15_design import exact_label_reference

    values = np.arange(1.0, 9.0)
    labels, oracle, cutoff = exact_label_reference(values, 4, 0.99)
    assert cutoff == pytest.approx(7.93)
    actual = []
    for indices in labels:
        mask = np.zeros(8, dtype=bool)
        mask[list(indices)] = True
        raw = WinsorRawState(
            metric="m",
            study_id="e",
            missingness="error",
            quantile=0.99,
            arms=(
                RawArm(group_id="C", values=tuple(values[mask])),
                RawArm(group_id="T", values=tuple(values[~mask])),
            ),
        )
        test = conditional_permutation_test(raw, "C", "T")
        assert test.exact and test.assignments == 70
        assert test.null_kind == "raw_distribution_exchangeability"
        assert WinsorPermutationTest.model_validate_json(test.model_dump_json()) == test
        actual.append(test.p_value)
    assert actual == pytest.approx(oracle)
    assert sum(p <= 0.05 for p in actual) == 2
    payload = raw.model_dump()
    for arm in payload["arms"]:
        arm["values"] = (1, 1, 1, 1)
    assert (
        conditional_permutation_test(WinsorRawState.model_validate(payload), "C", "T").p_value == 1
    )


@pytest.mark.slow
def test_frame_reference_survives_source_mutation_and_close():
    import pandas as pd

    from increment import readouts
    from increment.frame import from_unit_summary
    from increment.results import LiftEstimate

    frame = pd.DataFrame(
        {
            "u": range(12),
            "g": ["C"] * 4 + ["T"] * 5 + ["O"] * 3,
            "revenue": [1.0, 2.0, 3.0, 9.0, 2.0, 3.0, 5.0, 8.0, 12.0, 1.0, 4.0, 7.0],
        }
    )
    source = from_unit_summary(
        frame,
        unit="u",
        group="g",
        control="C",
        metrics=[{"name": "revenue", "winsorization": {"upper_percentile": 0.75}}],
    )
    frame.loc[:, "revenue"] = 999
    rows = readouts.run(source)
    assert {row.group_id for row in rows} == {"T", "O"}
    row = next(row for row in rows if row.group_id == "T")
    source.close()
    restored = LiftEstimate.model_validate_json(row.model_dump_json())
    assert restored.confidence_set is not None
    assert restored.confidence_set.raw.arm("C").values == (1, 2, 3, 9)
    assert restored.confidence_set.raw.arm("O").values == (1, 4, 7)
    assert restored.reintervalize(0.1) == row.reintervalize(0.1)


def test_empirical_population_center_is_not_mean_type7_bootstrap():
    # Exhaust all two-draw resamples of {1,3}; empirical p=.75 cutoff is 3.
    from itertools import product

    means = [
        float(np.minimum(draw, np.quantile(draw, 0.75, method="linear")).mean())
        for draw in product((1, 3), repeat=2)
    ]
    assert sum(means) / 4 == 1.875
    assert (1 + 3) / 2 == 2


@pytest.mark.slow
def test_relative_numeric_failure_does_not_hide_additive_availability():
    from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_reference
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.simulate.runner import _lift_outcome, _reduce_key
    from increment.tables import _liftestimate_to_row
    from increment.winsor import BootstrapReference

    raw = _state()
    payload = full_procedure_bootstrap_reference(raw, "C", "T").model_dump()
    payload["log_relative"]["roots"] = tuple(-10000.0 for _ in range(1999))
    reference = BootstrapReference.model_validate(payload)
    row = estimate_winsor_lift(raw, "C", "T", reference=reference)
    assert row.confidence_set is not None
    assert row.confidence_set.relative.upper.reason == "endpoint_unrepresentable"
    relative = _reduce_key([_lift_outcome(row.lift, confidence_set=row.confidence_set)], truth=0)
    additive = _reduce_key(
        [_lift_outcome(row.lift, confidence_set=row.confidence_set, winsor_scale="additive")],
        truth=0,
    )
    assert relative.attempted == additive.attempted == 1
    assert relative.point_estimable == additive.point_estimable == 1
    assert relative.interval_estimable == 0 and additive.interval_estimable == 1
    assert relative.coverage_unconditional == 0
    assert _liftestimate_to_row(row)["higher"] is None
    assert _liftestimate_to_row(row)["inference_method"] == reference.method
