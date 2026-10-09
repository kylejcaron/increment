"""Zero-inclusive pooled winsor inference: hurdle bootstrap, analytic interval, size route."""

from __future__ import annotations

import math
import statistics

import numpy as np
import pytest

SQRT_2PI = math.sqrt(2 * math.pi)


def _region(row):
    """The persisted confidence set of a winsor row, which is always present."""
    assert row.confidence_set is not None
    return row.confidence_set


def _zero_inflated(rng, n, zero_share, mu=0.0, sigma=1.0):
    y = rng.lognormal(mu, sigma, n)
    y[rng.random(n) < zero_share] = 0.0
    return y


def _raw(arms, quantile, *, method=None, stream=0):
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState

    spec = (
        WinsorInferenceSpec(stream=stream)
        if method is None
        else WinsorInferenceSpec(method=method, stream=stream)
    )
    return WinsorRawState(
        metric="revenue",
        study_id="zero",
        missingness="error",
        quantile=quantile,
        inference=spec,
        arms=tuple(RawArm(group_id=g, values=tuple(float(x) for x in v)) for g, v in arms.items()),
    )


def _three_arm_zero_inflated(seed=31415):
    rng = np.random.default_rng(seed)
    return {
        "C": _zero_inflated(rng, 2000, 0.60),
        "T": _zero_inflated(rng, 1500, 0.50, mu=0.1),
        "O": _zero_inflated(rng, 800, 0.70),
    }


def _brute_point(arms, quantile, control, treatment):
    from fractions import Fraction

    pooled = sorted(float(x) for v in arms.values() for x in v)
    rank = (len(pooled) - 1) * quantile
    lo, hi = math.floor(rank), math.ceil(rank)
    weight = Fraction(rank - lo)
    cutoff = (1 - weight) * Fraction(pooled[lo]) + weight * Fraction(pooled[hi])
    means = {
        g: sum((Fraction(min(float(x), float(cutoff))) for x in v), Fraction()) / len(v)
        for g, v in arms.items()
    }
    return (
        float(cutoff),
        float(means[treatment] / means[control] - 1),
        float(means[treatment] - means[control]),
    )


def _brute_influence(arms, quantile, control, treatment, cutoff):
    """Scalar loop over units: positive-part log kernel divided by the full pool size."""
    labels = sorted(arms)
    total = sum(len(arms[g]) for g in labels)
    density = 0.0
    for g in labels:
        logs = [math.log(float(x)) for x in arms[g] if x > 0]
        h = 1.06 * statistics.stdev(logs) * len(logs) ** -0.2
        for z in logs:
            u = (math.log(cutoff) - z) / h
            density += math.exp(-u * u / 2) / SQRT_2PI / (h * cutoff) / total
    means = {g: sum(min(float(x), cutoff) for x in arms[g]) / len(arms[g]) for g in labels}
    cdfs = {g: sum(float(x) <= cutoff for x in arms[g]) / len(arms[g]) for g in labels}
    ses = []
    for relative in (True, False):
        a = dict.fromkeys(labels, 0.0)
        a[control] = -1 / means[control] if relative else -1.0
        a[treatment] = 1 / means[treatment] if relative else 1.0
        derivative = sum(a[g] * (1 - cdfs[g]) for g in labels)
        variance = 0.0
        for g in labels:
            n = len(arms[g])
            b = derivative * (n / total) / density
            scores = [
                a[g] * (min(float(x), cutoff) - means[g]) + b * (cdfs[g] - (float(x) <= cutoff))
                for x in arms[g]
            ]
            variance += statistics.variance(scores) / n
        ses.append(math.sqrt(variance))
    return density, ses[0], ses[1]


def test_zero_inclusive_observed_statistics_match_scalar_brute_force():
    from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_reference

    arms = _three_arm_zero_inflated()
    raw = _raw(arms, 0.9)
    assert raw.executed_method == "positive-log-kernel-bootstrap-t-v1"
    reference = full_procedure_bootstrap_reference(raw, "C", "T")
    zero_counts = {p.group_id: p.zero_count for p in reference.pilots}
    assert zero_counts == {g: int(np.sum(v == 0)) for g, v in arms.items()}
    assert all(
        len(p.log_centers) == len(arms[g]) - zero_counts[g]
        for g, p in zip(sorted(arms), reference.pilots, strict=True)
    )
    cutoff, relative, additive = _brute_point(arms, 0.9, "C", "T")
    assert reference.observed_cutoff == cutoff
    assert math.expm1(reference.log_relative.point) == pytest.approx(relative, rel=1e-12)
    assert reference.additive.point == additive
    density, se_log, se_add = _brute_influence(arms, 0.9, "C", "T", cutoff)
    assert reference.log_relative.se == pytest.approx(se_log, rel=1e-12)
    assert reference.additive.se == pytest.approx(se_add, rel=1e-12)
    # The pooled density is the positive-part kernel density scaled by the positive share.
    positives = np.concatenate([v[v > 0] for v in arms.values()])
    positive_share = len(positives) / sum(len(v) for v in arms.values())
    f_plus = 0.0
    for g in sorted(arms):
        pos = arms[g][arms[g] > 0]
        logs = np.log(pos)
        h = 1.06 * logs.std(ddof=1) * len(pos) ** -0.2
        f_plus += (
            (len(pos) / len(positives))
            * float(np.exp(-0.5 * ((math.log(cutoff) - logs) / h) ** 2).sum())
            / (SQRT_2PI * len(pos) * h * cutoff)
        )
    assert density == pytest.approx(positive_share * f_plus, rel=1e-12)
    assert reference.failure_indices == ()


def test_hurdle_pilot_target_matches_numerical_integration():
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.stats import norm

    from increment.estimation._winsor_bootstrap import (
        fit_positive_log_pilot,
        pilot_parts,
        pilot_population_target,
    )

    raw = _raw(_three_arm_zero_inflated(), 0.9)
    pilots = fit_positive_log_pilot(raw)
    cutoff, ell, delta = pilot_population_target(pilots, 0.9, "C", "T")
    total = sum(p.zero_count + len(p.log_centers) for p in pilots)

    def pooled_cdf(c):
        return math.fsum(
            (p.zero_count + sum(norm.cdf((math.log(c) - z) / p.bandwidth) for z in p.log_centers))
            / total
            for p in pilots
        )

    expected_cutoff = brentq(lambda c: pooled_cdf(c) - 0.9, 1e-6, 1e4, xtol=1e-14, rtol=1e-14)
    assert cutoff == pytest.approx(expected_cutoff, rel=1e-10)
    means = {}
    for p in pilots:
        acc = 0.0
        for z in p.log_centers:
            boundary = (math.log(cutoff) - z) / p.bandwidth
            lower = quad(
                lambda u, z=z, h=p.bandwidth: math.exp(z + h * u) * norm.pdf(u),
                -12,
                boundary,
                epsabs=1e-13,
                epsrel=1e-12,
                limit=200,
            )[0]
            acc += lower + cutoff * norm.sf(boundary)
        means[p.group_id] = acc / (p.zero_count + len(p.log_centers))
        parts = pilot_parts(p, cutoff)
        assert parts[2] == pytest.approx(means[p.group_id], rel=1e-9)
        assert parts[0] == pytest.approx(
            (
                p.zero_count
                + sum(norm.cdf((math.log(cutoff) - z) / p.bandwidth) for z in p.log_centers)
            )
            / (p.zero_count + len(p.log_centers)),
            rel=1e-12,
        )
    assert ell == pytest.approx(math.log(means["T"] / means["C"]), abs=1e-9)
    assert delta == pytest.approx(means["T"] - means["C"], rel=1e-9)


def test_first_replicate_resamples_the_atom_and_matches_scalar_recomputation():
    from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_reference

    rng = np.random.default_rng(2024)
    arms = {"C": _zero_inflated(rng, 300, 0.6), "T": _zero_inflated(rng, 250, 0.5, mu=0.1)}
    raw = _raw(arms, 0.85, stream=5)
    reference = full_procedure_bootstrap_reference(raw, "C", "T")
    draws = {}
    for g, pilot in enumerate(reference.pilots):
        n = pilot.zero_count + len(pilot.log_centers)
        centers = np.random.Generator(
            np.random.PCG64DXSM(np.random.SeedSequence(1729, spawn_key=(5, g, 0)))
        )
        noise = np.random.Generator(
            np.random.PCG64DXSM(np.random.SeedSequence(1729, spawn_key=(5, g, 1)))
        )
        indices = centers.integers(n, size=(64, n))[0]
        eps = noise.standard_normal((64, n))[0]
        positive = indices >= pilot.zero_count
        values = np.zeros(n)
        values[positive] = np.exp(
            np.asarray(pilot.log_centers)[indices[positive] - pilot.zero_count]
            + pilot.bandwidth * eps[positive]
        )
        draws[pilot.group_id] = values
    assert all(int(np.sum(v == 0)) > 0 for v in draws.values())
    cutoff, relative, delta = _brute_point(draws, 0.85, "C", "T")
    _, se_log, se_add = _brute_influence(draws, 0.85, "C", "T", cutoff)
    expected_log_root = (math.log1p(relative) - reference.log_relative.pilot_target) / se_log
    expected_additive_root = (delta - reference.additive.pilot_target) / se_add
    assert reference.log_relative.roots[0] == pytest.approx(expected_log_root, rel=1e-9)
    assert reference.additive.roots[0] == pytest.approx(expected_additive_root, rel=1e-9)


def _atom_arms(rng, zeros_c, positives_c, zeros_t, positives_t):
    return {
        "C": np.concatenate([np.zeros(zeros_c), rng.lognormal(0.0, 1.0, positives_c)]),
        "T": np.concatenate([np.zeros(zeros_t), rng.lognormal(0.0, 1.0, positives_t)]),
    }


@pytest.mark.parametrize("method", [None, "influence-normal-v1"], ids=["routed", "analytic"])
@pytest.mark.parametrize(
    ("builder", "quantile", "code"),
    [
        (lambda rng: _atom_arms(rng, 1985, 15, 1990, 10), 0.99, "cutoff_in_zero_atom"),
        (lambda rng: _atom_arms(rng, 50, 50, 50, 50), 0.5, "cutoff_in_zero_atom"),
        (lambda rng: {"C": [0.0, 0.0, 1.0], "T": [1.0, 2.0, 3.0]}, 0.5, "pilot_degenerate"),
        (lambda rng: {"C": [0.0, 0.0, 0.0], "T": [1.0, 2.0, 3.0]}, 0.5, "pilot_degenerate"),
        (
            lambda rng: {"C": [1.0, 2.0, 3.0], "T": [1.0, 2.0, 4.0], "O": [0.0, 0.0, 0.0]},
            0.5,
            "pilot_degenerate",
        ),
        (lambda rng: {"C": [-1.0, 1.0, 2.0], "T": [1.0, 2.0, 3.0]}, 0.5, "pilot_negative_outcome"),
    ],
    ids=[
        "zero-share-above-q",
        "cutoff-straddles-atom",
        "one-positive",
        "all-zero-contrast",
        "all-zero-pool-arm",
        "negative",
    ],
)
def test_atom_and_sign_refusals_carry_stable_codes(method, builder, quantile, code):
    from increment.errors import CodedError
    from increment.estimation.winsor import estimate_winsor_lift

    arms = builder(np.random.default_rng(7))
    with pytest.raises(CodedError) as error:
        estimate_winsor_lift(_raw(arms, quantile, method=method), "C", "T")
    assert error.value.code == "estimation.winsor." + code


def test_cutoff_straddle_is_the_refusal_boundary():
    """One more positive moves the lower type-7 neighbour out of the atom."""
    from increment.errors import CodedError
    from increment.estimation.winsor import estimate_winsor_lift

    rng = np.random.default_rng(11)
    # N = 200, q = 0.5: rank 99.5, neighbours 99 and 100. Exactly 100 zeros straddle.
    with pytest.raises(CodedError) as error:
        estimate_winsor_lift(_raw(_atom_arms(rng, 50, 50, 50, 50), 0.5), "C", "T")
    assert error.value.code == "estimation.winsor.cutoff_in_zero_atom"
    row = estimate_winsor_lift(_raw(_atom_arms(rng, 49, 51, 50, 50), 0.5), "C", "T")
    assert row.confidence_set is not None
    assert row.confidence_set.method == "positive-log-kernel-bootstrap-t-v1"


def test_near_boundary_atom_replicates_fail_and_the_interval_is_unavailable():
    from scipy.stats import binom

    from increment.estimation._winsor_bootstrap import (
        bootstrap_confidence_set,
        full_procedure_bootstrap_reference,
    )

    rng = np.random.default_rng(2718)
    arms = {"C": _zero_inflated(rng, 1000, 0.985), "T": _zero_inflated(rng, 1000, 0.985)}
    raw = _raw(arms, 0.99)
    reference = full_procedure_bootstrap_reference(raw, "C", "T")
    pmf = np.array([1.0])
    for pilot in reference.pilots:
        n = pilot.zero_count + len(pilot.log_centers)
        pmf = np.convolve(pmf, binom.pmf(np.arange(n + 1), n, pilot.zero_count / n))
    threshold = math.floor((2000 - 1) * 0.99)
    atom_probability = float(pmf[threshold + 1 :].sum())
    expected = 1999 * atom_probability
    assert expected > 10
    assert len(reference.failure_indices) == pytest.approx(expected, abs=4 * math.sqrt(expected))
    region = bootstrap_confidence_set(reference, 0.05)
    assert region.relative.lower.status == "undefined"
    assert region.relative.lower.reason == "bootstrap_replicate_failure"
    assert region.additive.upper.reason == "bootstrap_replicate_failure"
    assert region.point is not None


def test_analytic_interval_uses_the_existing_influence_studentization():
    from scipy.special import ndtr, ndtri

    from increment.estimation.winsor import (
        estimate_winsor_lift,
        influence_reference,
        influence_studentization,
    )
    from increment.winsor import influence_p_value

    arms = _three_arm_zero_inflated()
    raw = _raw(arms, 0.9, method="influence-normal-v1")
    reference = influence_reference(raw, "C", "T")
    cutoff, relative, additive = _brute_point(arms, 0.9, "C", "T")
    density, se_log, se_add = _brute_influence(arms, 0.9, "C", "T", cutoff)
    assert reference.observed_cutoff == cutoff
    outcome_density = reference.scaled_density / reference.observed_cutoff
    assert outcome_density == pytest.approx(density, rel=1e-12)
    assert reference.log_relative.se == pytest.approx(se_log, rel=1e-12)
    assert reference.additive.se == pytest.approx(se_add, rel=1e-12)
    assert reference.log_relative.se == pytest.approx(
        math.sqrt(influence_studentization(raw, "C", "T", density=outcome_density)), rel=1e-12
    )
    assert reference.additive.se == pytest.approx(
        math.sqrt(
            influence_studentization(raw, "C", "T", density=outcome_density, contrast="difference")
        ),
        rel=1e-12,
    )
    row = estimate_winsor_lift(raw, "C", "T", alpha=0.1)
    region = _region(row)
    assert region.method == "influence-normal-v1"
    assert region.qualification == "pointwise_asymptotic_influence_v1"
    z = -float(ndtri(0.05))
    assert region.relative.lower.value == pytest.approx(
        math.expm1(reference.log_relative.point - z * se_log), rel=1e-12
    )
    assert region.relative.upper.value == pytest.approx(
        math.expm1(reference.log_relative.point + z * se_log), rel=1e-12
    )
    assert region.additive.lower.value == pytest.approx(additive - z * se_add, rel=1e-12)
    assert region.additive.upper.value == pytest.approx(additive + z * se_add, rel=1e-12)
    assert region.point == pytest.approx(relative, rel=1e-12)
    pivot = reference.log_relative.point / se_log
    assert row.p_value() == pytest.approx(2 * float(ndtr(-abs(pivot))), rel=1e-12)
    assert influence_p_value(reference, -1.0, relative=True) == 0.0
    p_value = row.p_value()
    assert p_value is not None and p_value < 0.05
    assert _region(row.reintervalize(0.01)).relative.lower.value < region.relative.lower.value


def test_analytic_reintervalization_preserves_smallest_positive_alpha():
    from increment.estimation.winsor import estimate_winsor_lift

    row = estimate_winsor_lift(
        _raw(_three_arm_zero_inflated(), 0.9, method="influence-normal-v1"), "C", "T"
    )
    alpha = math.ulp(0.0)
    region = _region(row.reintervalize(alpha))

    assert region.alpha == alpha
    for interval in (region.relative, region.additive):
        assert interval.lower.status == "finite"
        assert interval.upper.status == "finite"


@pytest.mark.parametrize(
    ("sizes", "quantile", "expected"),
    [
        ((9_999, 10_000), 0.99, "positive-log-kernel-bootstrap-t-v1"),
        ((10_000, 10_000), 0.99, "influence-normal-v1"),
        ((10_000, 10_000), 0.995, "influence-normal-v1"),
        ((10_000, 10_000), 0.999, "positive-log-kernel-bootstrap-t-v1"),
        ((50_000, 50_000), 0.999, "influence-normal-v1"),
        ((6_000, 7_000, 7_000), 0.99, "influence-normal-v1"),
    ],
)
def test_size_route_resolves_from_pool_size_and_quantile(sizes, quantile, expected):
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState

    arms = tuple(
        RawArm(group_id=label, values=tuple(float(x) for x in range(1, size + 1)))
        for label, size in zip("CTO", sizes, strict=False)
    )
    routed = WinsorRawState(
        metric="m", study_id="s", missingness="error", quantile=quantile, arms=arms
    )
    assert routed.inference.qualification == "pointwise_asymptotic_size_routed_v1"
    assert routed.executed_method == expected
    for explicit in ("positive-log-kernel-bootstrap-t-v1", "influence-normal-v1"):
        state = WinsorRawState(
            metric="m",
            study_id="s",
            missingness="error",
            quantile=quantile,
            inference=WinsorInferenceSpec(method=explicit),
            arms=arms,
        )
        assert state.executed_method == explicit


@pytest.mark.slow
def test_routed_row_records_the_executed_method_and_rejects_the_other_reference():
    from increment.errors import CodedError
    from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_reference
    from increment.estimation.winsor import estimate_winsor_lift, influence_reference
    from increment.results import LiftEstimate
    from increment.winsor import InfluenceReference, WinsorInferenceSpec

    rng = np.random.default_rng(777)
    arms = {"C": _zero_inflated(rng, 10_000, 0.9), "T": _zero_inflated(rng, 10_000, 0.9, mu=0.05)}
    raw = _raw(arms, 0.99)
    assert raw.inference.method == "pooled-size-route-v1"
    row = estimate_winsor_lift(raw, "C", "T")
    region = _region(row)
    reference = region.reference
    assert isinstance(reference, InfluenceReference)
    assert region.method == "influence-normal-v1"
    assert region.raw.inference.method == "pooled-size-route-v1"
    assert reference.spec == raw.inference
    restored = LiftEstimate.model_validate_json(row.model_dump_json())
    assert restored == row
    assert _region(restored.reintervalize(0.1)).alpha == 0.1
    assert restored.p_value() == row.p_value()
    assert row.winsor_control_n_upper is not None and row.winsor_treatment_n_upper is not None
    assert row.winsor_control_n_upper + row.winsor_treatment_n_upper > 0
    # A bootstrap reference built under an explicit request cannot stand in for the route.
    explicit = raw.model_copy(
        update={"inference": WinsorInferenceSpec(method="positive-log-kernel-bootstrap-t-v1")}
    )
    bootstrap = full_procedure_bootstrap_reference(explicit, "C", "T")
    with pytest.raises(CodedError) as error:
        estimate_winsor_lift(raw, "C", "T", reference=bootstrap)
    assert error.value.code == "estimation.winsor.pool_mismatch"
    with pytest.raises(CodedError) as error:
        estimate_winsor_lift(explicit, "C", "T", reference=influence_reference(raw, "C", "T"))
    assert error.value.code == "estimation.winsor.pool_mismatch"
    # The analytic reference refuses to describe a pool the route sends to the bootstrap.
    small = _raw({g: v[:500] for g, v in arms.items()}, 0.99)
    with pytest.raises(CodedError) as error:
        influence_reference(small, "C", "T")
    assert error.value.code == "estimation.winsor.invalid_state"


@pytest.mark.slow
def test_analytic_and_bootstrap_agree_at_a_large_pool_on_a_fixed_seed():
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.winsor import BootstrapReference, InfluenceReference

    rng = np.random.default_rng(777)
    arms = {"C": _zero_inflated(rng, 10_000, 0.9), "T": _zero_inflated(rng, 10_000, 0.9, mu=0.05)}
    analytic = _region(estimate_winsor_lift(_raw(arms, 0.99), "C", "T"))
    bootstrap = _region(
        estimate_winsor_lift(
            _raw(arms, 0.99, method="positive-log-kernel-bootstrap-t-v1"), "C", "T"
        )
    )
    analytic_reference, bootstrap_reference = analytic.reference, bootstrap.reference
    assert isinstance(analytic_reference, InfluenceReference)
    assert isinstance(bootstrap_reference, BootstrapReference)
    assert bootstrap_reference.failure_indices == ()
    assert analytic.point == bootstrap.point
    assert analytic.additive_point == bootstrap.additive_point
    se_log = analytic_reference.log_relative.se
    se_add = analytic_reference.additive.se
    assert bootstrap_reference.log_relative.se == pytest.approx(se_log, rel=1e-12)
    for analytic_end, bootstrap_end in (
        (analytic.relative.lower, bootstrap.relative.lower),
        (analytic.relative.upper, bootstrap.relative.upper),
    ):
        assert analytic_end.value is not None and bootstrap_end.value is not None
        assert abs(math.log1p(analytic_end.value) - math.log1p(bootstrap_end.value)) <= 0.1 * se_log
    for analytic_end, bootstrap_end in (
        (analytic.additive.lower, bootstrap.additive.lower),
        (analytic.additive.upper, bootstrap.additive.upper),
    ):
        assert analytic_end.value is not None and bootstrap_end.value is not None
        assert abs(analytic_end.value - bootstrap_end.value) <= 0.1 * se_add


def test_pilot_zero_count_is_persisted_and_archived_positive_payloads_still_load():
    from increment.errors import CodedError
    from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_reference
    from increment.winsor import BootstrapReference

    rng = np.random.default_rng(99)
    zero_raw = _raw({"C": _zero_inflated(rng, 60, 0.4), "T": _zero_inflated(rng, 60, 0.4)}, 0.8)
    reference = full_procedure_bootstrap_reference(zero_raw, "C", "T")
    payload = reference.model_dump(mode="json")
    assert payload["pilots"][0]["zero_count"] == reference.pilots[0].zero_count > 0
    assert BootstrapReference.model_validate(payload) == reference
    mutated = reference.model_dump(mode="json")
    mutated["pilots"][0]["zero_count"] -= 1
    with pytest.raises(CodedError) as error:
        BootstrapReference.model_validate(mutated)
    assert error.value.code == "estimation.winsor.pool_mismatch"
    positive_raw = _raw({"C": rng.lognormal(size=40), "T": rng.lognormal(size=40)}, 0.8)
    positive = full_procedure_bootstrap_reference(positive_raw, "C", "T")
    archived = positive.model_dump(mode="json")
    for pilot in archived["pilots"]:
        assert pilot.pop("zero_count") == 0
    assert BootstrapReference.model_validate(archived) == positive
    # An archived reference names the bootstrap explicitly rather than the size route.
    for spec in (archived["spec"], archived["raw"]["inference"]):
        spec["method"] = "positive-log-kernel-bootstrap-t-v1"
        spec["qualification"] = "pointwise_asymptotic_model_conditioned_v1"
    explicit = BootstrapReference.model_validate(archived)
    assert explicit.log_relative == positive.log_relative
    assert explicit.spec.method == "positive-log-kernel-bootstrap-t-v1"


@pytest.mark.parametrize(
    ("values", "code"),
    [
        ((float("nan"), 1.0), "model.field.nonfinite"),
        ((1.0, float("inf")), "model.field.nonfinite"),
        ((), "model.field.length"),
        (("1.5", 2.0), "model.field.type"),
        ((None,), "model.field.type"),
    ],
)
def test_direct_outcome_tuple_construction_establishes_the_field_invariants(values, code):
    """Building the validated outcome tuple by hand is as strict as the field validator."""
    from increment.errors import CodedError
    from increment.winsor import RawArm, _ValidatedOutcomes

    with pytest.raises(CodedError) as direct:
        _ValidatedOutcomes(values)
    assert direct.value.code == code
    with pytest.raises(CodedError) as through_model:
        RawArm(group_id="C", values=_ValidatedOutcomes(values))
    assert through_model.value.code == code


def test_direct_outcome_tuple_construction_sorts_and_coerces_like_the_field():
    from increment.winsor import RawArm, _ValidatedOutcomes

    direct = _ValidatedOutcomes((2.0, 1.0, 3.0))
    assert tuple(direct) == (1.0, 2.0, 3.0)
    assert RawArm(group_id="C", values=direct).values == (1.0, 2.0, 3.0)
    assert RawArm(group_id="C", values=_ValidatedOutcomes([3, 1.5])).values == (1.5, 3.0)
    assert _ValidatedOutcomes(direct) is direct
    assert (
        RawArm(group_id="C", values=(2.0, 1.0)).values
        == RawArm(group_id="C", values=_ValidatedOutcomes((2.0, 1.0))).values
    )


def test_overflowing_positive_draw_fails_a_zero_bearing_replicate_like_a_positive_one():
    """Masking covers exactly the zero logs; an overflow still poisons the row."""
    from increment.estimation._winsor_bootstrap import full_procedure_statistics_many

    other = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    overflowing = np.array([[1.0, 2.0, np.inf, 3.0, 5.0, 0.5]])
    with_zeros = np.array([[0.0, 0.0, np.inf, 3.0, 5.0, 0.5]])
    positive_only = full_procedure_statistics_many((overflowing, other), 0.75, ((0, 1),))[0]
    zero_bearing = full_procedure_statistics_many(
        (with_zeros, other), 0.75, ((0, 1),), atoms=(True, False)
    )[0]
    assert not positive_only.valid()[0]
    assert not zero_bearing.valid()[0]
    healthy = full_procedure_statistics_many(
        (np.array([[0.0, 0.0, 2.0, 3.0, 5.0, 0.5]]), other), 0.75, ((0, 1),), atoms=(True, False)
    )[0]
    assert healthy.valid()[0]


def test_overflowing_bootstrap_draws_in_a_zero_bearing_arm_are_failed_roots():
    from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_reference

    rng = np.random.default_rng(5)
    control = np.concatenate([np.zeros(40), rng.lognormal(0.0, 1.0, 60)])
    treatment = np.concatenate([np.zeros(30), [1e300, 1e305, 1e308] * 10, rng.lognormal(0, 1, 40)])
    raw = _raw({"C": control, "T": treatment}, 0.8, method="positive-log-kernel-bootstrap-t-v1")
    reference = full_procedure_bootstrap_reference(raw, "C", "T")
    assert reference.failure_indices
    assert all(reference.log_relative.roots[index] is None for index in reference.failure_indices)


def test_analytic_reference_is_scale_invariant_down_to_subnormal_outcomes():
    """The persisted density is dimensionless, so tiny outcomes stay representable."""
    from increment.estimation.winsor import estimate_winsor_lift

    rng = np.random.default_rng(6)
    base = {"C": rng.lognormal(0, 1, 6000), "T": rng.lognormal(0.05, 1, 6000)}
    tiny = {g: v * 1e-312 for g, v in base.items()}
    for scale, arms in ((1.0, base), (1e-312, tiny)):
        row = estimate_winsor_lift(_raw(arms, 0.75, method="influence-normal-v1"), "C", "T")
        region = _region(row)
        assert region.method == "influence-normal-v1"
        assert region.relative.lower.value is not None and region.relative.upper.value is not None
        if scale == 1.0:
            expected = (region.relative.lower.value, region.relative.upper.value)
        else:
            assert region.relative.lower.value == pytest.approx(expected[0], rel=1e-6)
            assert region.relative.upper.value == pytest.approx(expected[1], rel=1e-6)
    alternating = tuple([1e-312, 2e-312] * 5000)
    region = _region(
        estimate_winsor_lift(
            _raw({"C": alternating, "T": alternating}, 0.75, method="influence-normal-v1"), "C", "T"
        )
    )
    assert region.additive.lower.value is not None and region.additive.upper.value is not None
    assert region.point == 0.0


@pytest.mark.parametrize("method", ["influence-normal-v1", "positive-log-kernel-bootstrap-t-v1"])
def test_restored_references_apply_the_same_arm_refusals_as_fresh_construction(method):
    """Editing a saved pool into an inadmissible one refuses on load with the fresh code."""
    from increment.errors import CodedError
    from increment.estimation.winsor import build_winsor_references, estimate_winsor_lift
    from increment.winsor import BootstrapReference, InfluenceReference, WinsorRawState

    arms = {"C": (1.0, 2.0, 3.0), "T": (1.0, 2.0, 4.0), "O": (0.1, 0.2, 0.3)}
    raw = _raw(arms, 0.9, method=method)
    reference = build_winsor_references(raw, "C", ("T",))["T"]
    model = InfluenceReference if method == "influence-normal-v1" else BootstrapReference
    assert model.model_validate(reference.model_dump(mode="json")) == reference
    payload = reference.model_dump(mode="json")
    arm_index = [arm["group_id"] for arm in payload["raw"]["arms"]].index("O")
    payload["raw"]["arms"][arm_index]["values"] = [0.0, 0.0, 0.0]
    with pytest.raises(CodedError) as restored:
        model.model_validate(payload)
    assert restored.value.code == "estimation.winsor.pilot_degenerate"
    with pytest.raises(CodedError) as fresh:
        estimate_winsor_lift(WinsorRawState.model_validate(payload["raw"]), "C", "T")
    assert fresh.value.code == "estimation.winsor.pilot_degenerate"
    payload = reference.model_dump(mode="json")
    payload["raw"]["arms"][arm_index]["values"] = [-1.0, 0.2, 0.3]
    with pytest.raises(CodedError) as negative:
        model.model_validate(payload)
    assert negative.value.code == "estimation.winsor.pilot_negative_outcome"
