"""Independent finite arithmetic and portable full-procedure references."""

import math

import numpy as np
import pytest


def test_threaded_bootstrap_sampling_is_bitwise_equal_and_keeps_arm_streams():
    from concurrent.futures import ThreadPoolExecutor

    from increment.estimation._winsor_bootstrap import (
        _PilotData,
        _sample_bootstrap_block,
    )

    pilots = [
        _PilotData("C", np.linspace(-1.0, 1.0, 200), 0.25),
        _PilotData("T", np.linspace(-2.0, 2.0, 800), 0.5),
    ]

    def run(executor):
        centers = (
            (
                np.random.Generator(np.random.PCG64DXSM(19)),
                np.random.Generator(np.random.PCG64DXSM(20)),
            ),
            (
                np.random.Generator(np.random.PCG64DXSM(29)),
                np.random.Generator(np.random.PCG64DXSM(30)),
            ),
        )
        log_buffers = tuple(np.empty(16 * len(pilot.log_centers)) for pilot in pilots)
        draw_buffers = tuple(np.empty_like(buffer) for buffer in log_buffers)
        noise_buffers = tuple(np.empty_like(buffer) for buffer in log_buffers)
        return _sample_bootstrap_block(
            pilots,
            tuple(pilot.log_centers for pilot in pilots),
            centers,
            log_buffers,
            draw_buffers,
            noise_buffers,
            16,
            executor=executor,
        )

    serial_draws, serial_logs = run(None)
    with ThreadPoolExecutor(max_workers=2) as executor:
        threaded_draws, threaded_logs = run(executor)

    for expected, actual in zip(serial_draws, threaded_draws, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for expected, actual in zip(serial_logs, threaded_logs, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("sizes", "quantile"), [((200, 800), 0.95), ((1000, 1000), 0.99)])
def test_threaded_statistics_match_b_grid_serial_bits(sizes, quantile):
    from concurrent.futures import ThreadPoolExecutor

    from increment.estimation._winsor_bootstrap import full_procedure_statistics_many

    rng = np.random.default_rng(917)
    samples = tuple(np.exp(rng.normal(size=(8, size))) for size in sizes)
    logs = tuple(np.log(sample) for sample in samples)
    serial = full_procedure_statistics_many(samples, quantile, ((0, 1),), log_samples=logs)[0]
    with ThreadPoolExecutor(max_workers=2) as executor:
        threaded = full_procedure_statistics_many(
            samples, quantile, ((0, 1),), log_samples=logs, _executor=executor
        )[0]
    assert np.array_equal(threaded.valid(), serial.valid())
    for field in ("cutoff", "log_relative", "additive", "log_se", "additive_se", "density_scaled"):
        np.testing.assert_array_equal(getattr(threaded, field), getattr(serial, field))


def test_threaded_statistics_match_serial_near_one_degeneracy():
    from concurrent.futures import ThreadPoolExecutor

    from increment.estimation._winsor_bootstrap import (
        _realized_draw_and_logs,
        full_procedure_statistics_many,
    )

    control_logs = np.resize(np.asarray([1e-17, 2e-17, 2.3e-16, 2.6e-16]), 240)[None, :]
    treatment_logs = control_logs + 16 * np.finfo(float).eps
    with np.errstate(all="ignore"):
        realized = tuple(_realized_draw_and_logs(logs) for logs in (control_logs, treatment_logs))
    samples = tuple(draw for draw, _ in realized)
    realized_logs = tuple(logs for _, logs in realized)
    serial = full_procedure_statistics_many(samples, 0.5, ((0, 1),), log_samples=realized_logs)[0]
    with ThreadPoolExecutor(max_workers=2) as executor:
        threaded = full_procedure_statistics_many(
            samples, 0.5, ((0, 1),), log_samples=realized_logs, _executor=executor
        )[0]
    assert np.array_equal(threaded.valid(), serial.valid())
    assert not threaded.valid()[0]
    for field in ("cutoff", "log_relative", "additive", "log_se", "additive_se", "density_scaled"):
        np.testing.assert_array_equal(getattr(threaded, field), getattr(serial, field))


def test_threaded_density_exception_keeps_serial_arm_precedence(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    import increment.estimation._winsor_bootstrap as bootstrap
    from increment.errors import CodedError

    def fail_by_arm(log_cutoff, values, bandwidth, total, work):
        del log_cutoff, bandwidth, total, work
        if values[0, 0] == 1.0:
            raise CodedError("first arm failure", code="test.first_arm", context={})
        time.sleep(0.01)
        raise CodedError("second arm failure", code="test.second_arm", context={})

    monkeypatch.setattr(bootstrap, "_scaled_log_density_arm", fail_by_arm)
    logs = (np.asarray([[1.0]]), np.asarray([[2.0]]))
    bandwidths = (np.ones(1), np.ones(1))

    def failure_code(executor):
        with pytest.raises(CodedError) as raised:
            bootstrap._scaled_log_density(np.ones(1), logs, bandwidths, 2, executor=executor)
        return raised.value.code

    serial_code = failure_code(None)
    with ThreadPoolExecutor(max_workers=2) as executor:
        threaded_code = failure_code(executor)
    assert serial_code == threaded_code == "test.first_arm"


@pytest.mark.slow
def test_three_arm_threaded_references_match_serial_bytes_and_reuse_density_workspaces(
    monkeypatch,
):
    from collections import Counter
    from threading import Lock

    import increment.estimation._winsor_bootstrap as bootstrap
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState

    rng = np.random.default_rng(381)
    arms = tuple(
        RawArm(
            group_id=group,
            values=tuple(np.exp(rng.normal(location, 0.45, size=2_000)).tolist()),
        )
        for group, location in (("C", 0.0), ("T1", 0.04), ("T2", -0.03))
    )
    raw = WinsorRawState(
        metric="revenue",
        study_id="threaded-parity",
        missingness="error",
        quantile=0.95,
        inference=WinsorInferenceSpec(stream=381),
        arms=arms,
    )
    assert sum(len(arm.values) for arm in arms) > 4_000

    monkeypatch.setattr(bootstrap, "_THREAD_POOL_MIN_BLOCK_ELEMENTS", 1_000_000_000)
    serial = bootstrap.full_procedure_bootstrap_references(raw, "C", ("T1", "T2"))

    original_density_arm = bootstrap._scaled_log_density_arm
    workspace_addresses = []
    workspace_lock = Lock()

    def record_density_workspace(log_cutoff, values, bandwidth, total, work):
        with workspace_lock:
            workspace_addresses.append(int(work.ctypes.data))
        return original_density_arm(log_cutoff, values, bandwidth, total, work)

    monkeypatch.setattr(bootstrap, "_scaled_log_density_arm", record_density_workspace)
    monkeypatch.setattr(bootstrap, "_THREAD_POOL_MIN_BLOCK_ELEMENTS", 1)
    threaded = bootstrap.full_procedure_bootstrap_references(raw, "C", ("T1", "T2"))

    for treatment in ("T1", "T2"):
        expected = serial[treatment]
        actual = threaded[treatment]
        assert actual == expected
        for attribute in ("log_relative", "additive"):
            expected_root = getattr(expected, attribute)
            actual_root = getattr(actual, attribute)
            assert (
                np.asarray(
                    (expected_root.point, expected_root.pilot_target, expected_root.se),
                    dtype=np.float64,
                ).tobytes()
                == np.asarray(
                    (actual_root.point, actual_root.pilot_target, actual_root.se),
                    dtype=np.float64,
                ).tobytes()
            )
            assert tuple(value for value in actual_root.roots if value is None) == tuple(
                value for value in expected_root.roots if value is None
            )
            expected_values = np.asarray(
                [value for value in expected_root.roots if value is not None], dtype=np.float64
            )
            actual_values = np.asarray(
                [value for value in actual_root.roots if value is not None], dtype=np.float64
            )
            assert actual_values.tobytes() == expected_values.tobytes()
    assert len(serial["T1"].log_relative.roots) == 1_999

    reused = Counter(workspace_addresses)
    reused = {address: count for address, count in reused.items() if count > 1}
    assert len(reused) == 3
    assert set(reused.values()) == {32}


@pytest.mark.slow
def test_nested_non_main_thread_bootstrap_matches_serial_references(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import increment.estimation._winsor_bootstrap as bootstrap

    raw = _state(stream=62)
    monkeypatch.setattr(bootstrap, "_THREAD_POOL_MIN_BLOCK_ELEMENTS", 1_000_000_000)
    serial = bootstrap.full_procedure_bootstrap_references(raw, "C", ("T", "O"))
    monkeypatch.setattr(bootstrap, "_THREAD_POOL_MIN_BLOCK_ELEMENTS", 1)
    with ThreadPoolExecutor(max_workers=1) as outer:
        nested = outer.submit(
            bootstrap.full_procedure_bootstrap_references, raw, "C", ("T", "O")
        ).result(timeout=60)
    assert nested == serial
    for treatment in ("T", "O"):
        for attribute in ("log_relative", "additive"):
            expected = getattr(serial[treatment], attribute)
            actual = getattr(nested[treatment], attribute)
            assert tuple(i for i, value in enumerate(actual.roots) if value is None) == tuple(
                i for i, value in enumerate(expected.roots) if value is None
            )
            assert (
                np.asarray(
                    [value for value in actual.roots if value is not None], dtype=np.float64
                ).tobytes()
                == np.asarray(
                    [value for value in expected.roots if value is not None], dtype=np.float64
                ).tobytes()
            )


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


def test_revalidating_numeric_payload_models_reuses_finite_float_tuples():
    from increment.winsor import PositiveLogPilot, RawArm

    values = tuple(float(index) for index in range(4096))
    arm = RawArm(group_id="C", values=values)
    pilot = PositiveLogPilot(group_id="C", log_centers=values, bandwidth=1.0)

    assert RawArm.model_validate(arm).values is arm.values
    assert PositiveLogPilot.model_validate(pilot).log_centers is pilot.log_centers


def test_pilot_work_centers_are_reused_as_readonly_numpy_arrays():
    import increment.estimation._winsor_bootstrap as bootstrap

    raw = _state()
    work = bootstrap._fit_positive_log_pilot_data(raw)
    fitted = bootstrap.fit_positive_log_pilot(raw)

    assert all(isinstance(pilot.log_centers, np.ndarray) for pilot in work)
    for actual, expected in zip(work, fitted, strict=True):
        assert not actual.log_centers.flags.writeable
        np.testing.assert_array_equal(actual.log_centers, expected.log_centers)
        assert actual.bandwidth == expected.bandwidth


def test_multi_contrast_reuses_pool_and_preserves_references_across_subblocks(monkeypatch):
    import increment.estimation._winsor_bootstrap as bootstrap

    raw = _state(stream=7)
    expected = {
        treatment: bootstrap.full_procedure_bootstrap_reference(raw, "C", treatment)
        for treatment in ("T", "O")
    }
    monkeypatch.setattr(bootstrap, "_WORKING_SET_ELEMENTS", 1)
    actual = bootstrap.full_procedure_bootstrap_references(raw, "C", ("T", "O"))
    assert actual == expected


def test_multi_contrast_reuses_one_exact_observed_point_summary(monkeypatch):
    import increment.estimation._winsor_bootstrap as bootstrap
    from increment.winsor import WinsorRawState

    raw = _state(stream=8)
    original = WinsorRawState._exact_point_summary
    calls = 0

    def count_summaries(state, **kwargs):
        nonlocal calls
        calls += 1
        return original(state, **kwargs)

    monkeypatch.setattr(WinsorRawState, "_exact_point_summary", count_summaries)
    bootstrap.full_procedure_bootstrap_references(raw, "C", ("T", "O"))
    assert calls == 1


def test_kernel_reuses_readonly_pilot_arrays_across_bootstrap_replicates(monkeypatch):
    import increment.estimation._winsor_bootstrap as bootstrap

    raw = _state(stream=8)
    original_fit = bootstrap._fit_positive_log_pilot_data
    center_array_ids = set()

    def record_centers(raw_state):
        pilots = original_fit(raw_state)
        center_array_ids.update(id(pilot.log_centers) for pilot in pilots)
        return pilots

    monkeypatch.setattr(bootstrap, "_fit_positive_log_pilot_data", record_centers)
    original_asarray = bootstrap.np.asarray
    conversions = 0

    def count_center_coercions(value, *args, **kwargs):
        nonlocal conversions
        if id(value) in center_array_ids:
            conversions += 1
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(bootstrap.np, "asarray", count_center_coercions)
    bootstrap.full_procedure_bootstrap_references(raw, "C", ("T", "O"))
    assert conversions == 0


@pytest.mark.parametrize(
    ("total_units", "expected_rows"),
    [(2_000, 64), (16_384, 64), (16_385, 32), (20_000, 32), (2_000_000, 1)],
)
def test_block_rows_respects_working_set_and_logical_chunk(total_units, expected_rows):
    from increment.estimation._winsor_bootstrap import _block_rows

    assert _block_rows(total_units) == expected_rows


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


def test_large_pool_score_moments_match_explicit_scores_near_one():
    from increment.estimation._winsor_bootstrap import full_procedure_statistics_many

    lower = np.nextafter(1.0, 0.0)
    lower2 = np.nextafter(lower, 0.0)
    raw = (
        np.tile(np.asarray([1.0, lower, lower2]), 1_000),
        np.tile(np.asarray([1.0, lower]), 1_500),
    )
    samples = tuple(values[None, :] for values in raw)

    actual = full_procedure_statistics_many(samples, 0.75, ((0, 1),))[0]
    expected = _independent_scores(raw, 0.75, 0, 1)

    assert actual.valid()[0]
    np.testing.assert_allclose(
        (actual.cutoff[0], actual.log_relative[0], actual.additive[0]),
        expected[:3],
        rtol=1e-12,
        atol=1e-12,
    )
    for observed_se, expected_se in zip(
        (actual.log_se[0], actual.additive_se[0]), expected[3:], strict=True
    ):
        assert observed_se > 0 and expected_se > 0
        assert observed_se == pytest.approx(expected_se, rel=1e-12, abs=0.0)


def test_large_pool_tiny_additive_variance_preserves_legacy_degeneracy():
    from increment.errors import CodedError
    from increment.estimation._winsor_bootstrap import (
        full_procedure_bootstrap_references,
        full_procedure_statistics_many,
    )
    from increment.winsor import RawArm, WinsorRawState

    control = np.tile([1e-200, 2e-200], 1001)
    treatment = np.tile([1e-200, 2e-200], 1001)
    other = np.tile([1.0, 2.0], 1001)
    samples = (control[None, :], treatment[None, :], other[None, :])

    actual = full_procedure_statistics_many(samples, 0.75, ((0, 1),))[0]
    expected = _independent_scores((control, treatment, other), 0.75, 0, 1)

    assert expected[4] == 0.0
    assert actual.additive_se[0] == 0.0
    assert not actual.valid()[0]
    raw = WinsorRawState(
        metric="m",
        study_id="underflow-degeneracy",
        missingness="error",
        quantile=0.75,
        arms=(
            RawArm(group_id="C", values=tuple(control)),
            RawArm(group_id="T", values=tuple(treatment)),
            RawArm(group_id="O", values=tuple(other)),
        ),
    )
    with pytest.raises(CodedError) as error:
        full_procedure_bootstrap_references(raw, "C", ("T",))
    assert error.value.code == "estimation.winsor.studentization_degenerate"


def test_large_pool_score_moments_preserve_tiny_arm_variance():
    from increment.estimation._winsor_bootstrap import full_procedure_statistics_many

    control = np.tile([1e-200, 2e-200], 1001)[None, :]
    treatment = np.tile([1.0, 2.0], 1001)[None, :]
    actual = full_procedure_statistics_many((control, treatment), 0.75, ((0, 1),))[0]
    expected = _independent_scores((control[0], treatment[0]), 0.75, 0, 1)

    assert np.isfinite(actual.log_se[0])
    assert actual.log_se[0] == pytest.approx(expected[3], rel=1e-12)
    assert actual.additive_se[0] == pytest.approx(expected[4], rel=1e-12)


def test_extreme_score_coefficients_fall_back_before_quadratic_overflow(monkeypatch):
    from increment.estimation import _winsor_bootstrap as kernel

    samples = (
        np.tile(np.asarray([1.0, 1.0, 2.0]), 833)[None, :],
        np.tile(np.asarray([1.0, 2.0]), 1_250)[None, :],
    )
    monkeypatch.setattr(
        kernel,
        "_scaled_log_density",
        lambda log_cutoff, *_args, **_kwargs: np.full_like(log_cutoff, 1e-200),
    )

    actual = kernel.full_procedure_statistics_many(samples, 0.5, ((0, 1),))[0]

    assert not actual.valid()[0]
    assert np.isinf(actual.log_se[0])


def test_tail_type7_cutoff_matches_numpy_with_ties_and_nonfinite_rows():
    from increment.estimation._winsor_bootstrap import _tail_type7_cutoff

    rng = np.random.default_rng(24)
    sample = rng.lognormal(0.0, 0.5, size=(4, 64))
    sample[1] = 3.0
    sample[2, :5] = (0.0, np.nextafter(0.0, 1.0), 1e-250, 1e250, np.inf)
    sample[3, 7] = np.nan
    samples = (sample[:, :23], sample[:, 23:])
    pooled = np.concatenate(samples, axis=1)
    for quantile in (0.95, 0.99):
        expected = np.quantile(pooled, quantile, axis=1, method="linear")
        lower_bound = float(np.quantile(pooled[0], quantile, method="linear")) * 0.9
        actual = _tail_type7_cutoff(samples, quantile, lower_bound)
        np.testing.assert_array_equal(actual, expected)


def test_statistics_many_accepts_precomputed_log_samples_exactly():
    from increment.estimation._winsor_bootstrap import full_procedure_statistics_many

    samples = (
        np.asarray([[1.0, 2.0, 4.0], [1.0, 8.0, 12.0]]),
        np.asarray([[2.0, 3.0, 8.0, 10.0], [2.0, 3.0, 4.0, 5.0]]),
        np.asarray([[3.0, 9.0], [1.0, 6.0]]),
    )
    contrasts = ((0, 1), (0, 2))
    expected = full_procedure_statistics_many(samples, 0.75, contrasts)
    actual = full_procedure_statistics_many(
        samples, 0.75, contrasts, log_samples=tuple(np.log(y) for y in samples)
    )
    for expected_stats, actual_stats in zip(expected, actual, strict=True):
        for name in (
            "cutoff",
            "log_relative",
            "additive",
            "log_se",
            "additive_se",
            "density_scaled",
        ):
            np.testing.assert_array_equal(
                getattr(actual_stats, name), getattr(expected_stats, name)
            )


@pytest.mark.parametrize(
    ("control_draw", "control_log"),
    [(np.inf, 710.0), (0.0, -800.0)],
    ids=("overflow", "underflow"),
)
def test_precomputed_draw_logs_preserve_realized_extreme_failure_masks(control_draw, control_log):
    from increment.estimation._winsor_bootstrap import (
        _realized_draw_and_logs,
        full_procedure_statistics_many,
    )

    latent_draw_logs = (
        np.asarray([[0.0, control_log]]),
        np.asarray([[2.0, 3.0]]),
    )
    with np.errstate(all="ignore"):
        realized = tuple(_realized_draw_and_logs(log_draw) for log_draw in latent_draw_logs)
    samples = tuple(draw for draw, _ in realized)
    realized_logs = tuple(logs for _, logs in realized)
    assert samples[0][0, 1] == control_draw
    old_refit = full_procedure_statistics_many(samples, 0.5, ((0, 1),))[0]
    cached_refit = full_procedure_statistics_many(
        samples, 0.5, ((0, 1),), log_samples=realized_logs
    )[0]
    np.testing.assert_array_equal(cached_refit.valid(), old_refit.valid())
    assert not old_refit.valid()[0]
    assert not cached_refit.valid()[0]


def test_normal_draw_logs_match_the_realized_sample():
    from increment.estimation._winsor_bootstrap import (
        _realized_draw_and_logs,
        full_procedure_statistics_many,
    )

    rounded_logs = np.asarray([[1e-18, 2e-18], [-0.5, np.nextafter(-0.5, np.inf)]])
    with np.errstate(all="ignore"):
        draws, realized_logs = _realized_draw_and_logs(rounded_logs)
    np.testing.assert_array_equal(realized_logs, np.log(draws))
    assert draws[1, 0] == draws[1, 1]
    assert realized_logs[1, 0] == realized_logs[1, 1]

    samples = (draws[1:2], np.asarray([[1.0, 2.0]]))
    old_refit = full_procedure_statistics_many(samples, 0.75, ((0, 1),))[0]
    cached_refit = full_procedure_statistics_many(
        samples,
        0.75,
        ((0, 1),),
        log_samples=(realized_logs[1:2], np.log(samples[1])),
    )[0]
    np.testing.assert_array_equal(cached_refit.valid(), old_refit.valid())
    assert not old_refit.valid()[0]


def test_nonconstant_near_one_realized_draws_preserve_legacy_failure_mask():
    from increment.estimation._winsor_bootstrap import (
        _realized_draw_and_logs,
        full_procedure_statistics_many,
    )

    control_logs = np.resize(np.asarray([1e-17, 2e-17, 2.3e-16, 2.6e-16]), 240)[None, :]
    treatment_logs = control_logs + 16 * np.finfo(float).eps
    with np.errstate(all="ignore"):
        realized = tuple(_realized_draw_and_logs(logs) for logs in (control_logs, treatment_logs))
    samples = tuple(draw for draw, _ in realized)
    realized_logs = tuple(logs for _, logs in realized)

    legacy = full_procedure_statistics_many(samples, 0.5, ((0, 1),))[0]
    candidate = full_procedure_statistics_many(samples, 0.5, ((0, 1),), log_samples=realized_logs)[
        0
    ]

    np.testing.assert_array_equal(candidate.valid(), legacy.valid())
    assert not legacy.valid()[0]


def test_realized_draw_logs_are_independent_of_execution_blocking():
    from increment.estimation._winsor_bootstrap import _realized_draw_and_logs

    control_logs = np.resize(np.asarray([1e-17, 2e-17, 2.3e-16, 2.6e-16]), 240)
    control_logs = np.stack((control_logs, np.linspace(-2.0, 2.0, 240)))
    control_logs[1, -1] = 701.0
    with np.errstate(all="ignore"):
        _, full_logs = _realized_draw_and_logs(control_logs.copy())
        _, blocked_logs = _realized_draw_and_logs(control_logs[:1].copy())
        _, blocked_tail = _realized_draw_and_logs(control_logs[1:].copy())

    np.testing.assert_array_equal(full_logs, np.concatenate((blocked_logs, blocked_tail)))


def test_near_one_realized_draws_preserve_rounding_degeneracy_mask():
    from increment.estimation._winsor_bootstrap import (
        _realized_draw_and_logs,
        full_procedure_statistics_many,
    )

    latent_draw_logs = (
        np.asarray([[1e-17, 2e-17, -1e-17, 2.3e-16, 2.6e-16, 0.1, 0.25]]),
        np.asarray([[-2e-17, 3e-17, 4e-17, 2.4e-16, 2.7e-16, 0.12, 0.32]]),
    )
    with np.errstate(all="ignore"):
        realized = tuple(_realized_draw_and_logs(logs) for logs in latent_draw_logs)
    samples = tuple(draw for draw, _ in realized)
    realized_logs = tuple(logs for _, logs in realized)
    old_logs = tuple(np.log(draw) for draw in samples)
    assert np.unique(samples[0][0, :5]).size < 5
    np.testing.assert_array_equal(realized_logs[0][0, :5], old_logs[0][0, :5])

    old_refit = full_procedure_statistics_many(samples, 0.75, ((0, 1),))[0]
    cached_refit = full_procedure_statistics_many(
        samples, 0.75, ((0, 1),), log_samples=realized_logs
    )[0]
    np.testing.assert_array_equal(cached_refit.valid(), old_refit.valid())
    for field in ("log_relative", "additive", "log_se", "additive_se", "density_scaled"):
        np.testing.assert_allclose(
            getattr(cached_refit, field), getattr(old_refit, field), rtol=1e-12, atol=1e-12
        )


def test_large_pool_cached_normal_logs_preserve_rounding_degeneracy_mask():
    from increment.estimation._winsor_bootstrap import (
        _realized_draw_and_logs,
        full_procedure_statistics_many,
    )

    control_logs = np.resize(np.asarray([-0.5, np.nextafter(-0.5, np.inf)]), 100_000)[None, :]
    treatment_logs = np.resize(np.asarray([0.0, math.log(2.0)]), 100_000)[None, :]
    with np.errstate(all="ignore"):
        realized = tuple(_realized_draw_and_logs(logs) for logs in (control_logs, treatment_logs))
    samples = tuple(draw for draw, _ in realized)
    cached_logs = tuple(logs for _, logs in realized)
    old_logs = tuple(np.log(draws) for draws in samples)

    old_statistics = full_procedure_statistics_many(samples, 0.75, ((0, 1),), log_samples=old_logs)[
        0
    ]
    cached_statistics = full_procedure_statistics_many(
        samples, 0.75, ((0, 1),), log_samples=cached_logs
    )[0]

    np.testing.assert_array_equal(cached_statistics.valid(), old_statistics.valid())
    assert not old_statistics.valid()[0]


def test_large_pool_cached_logs_preserve_mixed_rounding_reference():
    from increment.estimation._winsor_bootstrap import (
        _realized_draw_and_logs,
        full_procedure_statistics_many,
    )

    control_logs = np.resize(
        np.asarray([1e-17, 2e-17, -1e-17, 2.3e-16, 2.6e-16, 0.1, 0.25]), 100_000
    )[None, :]
    treatment_logs = np.resize(
        np.asarray([-2e-17, 3e-17, 4e-17, 2.4e-16, 2.7e-16, 0.12, 0.32]), 100_000
    )[None, :]
    realized = tuple(_realized_draw_and_logs(logs) for logs in (control_logs, treatment_logs))
    samples = tuple(draw for draw, _ in realized)
    cached_logs = tuple(logs for _, logs in realized)
    old_logs = tuple(np.log(draws) for draws in samples)
    assert np.unique(samples[0]).size < 7

    old_statistics = full_procedure_statistics_many(samples, 0.75, ((0, 1),), log_samples=old_logs)[
        0
    ]
    cached_statistics = full_procedure_statistics_many(
        samples, 0.75, ((0, 1),), log_samples=cached_logs
    )[0]

    np.testing.assert_array_equal(cached_statistics.valid(), old_statistics.valid())
    assert old_statistics.valid()[0]
    for field in ("log_relative", "additive", "log_se", "additive_se", "density_scaled"):
        np.testing.assert_allclose(
            getattr(cached_statistics, field),
            getattr(old_statistics, field),
            rtol=1e-12,
            atol=1e-12,
        )


def test_realized_draw_logs_write_into_reused_draw_buffer():
    from increment.estimation._winsor_bootstrap import _realized_draw_and_logs

    log_draw = np.asarray([[0.2, -0.3, 1.1], [2.0, -1.0, 0.5]])
    original = log_draw.copy()
    expected_draw = np.exp(original)
    expected_logs = np.log(expected_draw)
    draw_buffer = np.empty_like(log_draw)

    with np.errstate(all="ignore"):
        draws, realized_logs = _realized_draw_and_logs(log_draw, draw_output=draw_buffer)

    assert draws is draw_buffer
    assert realized_logs is log_draw
    np.testing.assert_array_equal(draws, expected_draw)
    np.testing.assert_array_equal(realized_logs, expected_logs)


def test_pivot_block_persistence_keeps_exact_values_and_failure_indices():
    from increment.estimation._winsor_bootstrap import _append_pivot_block

    log_roots: list[list[float | None]] = [[]]
    additive_roots: list[list[float | None]] = [[]]
    failures: list[list[int]] = [[]]
    pivots = [
        (
            np.asarray([1.0, np.inf, 3.0]),
            np.asarray([0.1, 0.2, np.nan]),
            np.asarray([True, True, False]),
        )
    ]

    _append_pivot_block(log_roots, additive_roots, failures, pivots, first_index=10)

    assert log_roots == [[1.0, None, None]]
    assert additive_roots == [[0.1, None, None]]
    assert failures == [[11, 12]]


def test_large_finite_draw_logs_match_the_realized_values():
    from increment.estimation._winsor_bootstrap import _realized_draw_and_logs

    latent_logs = np.linspace(-10.0, 10.0, 100_000)
    expected_draws = np.exp(latent_logs)
    expected_logs = np.log(expected_draws)
    with np.errstate(all="ignore"):
        draws, realized_logs = _realized_draw_and_logs(latent_logs)
    assert np.all(np.isfinite(draws) & (draws > 0))
    np.testing.assert_array_equal(draws, expected_draws)
    np.testing.assert_array_equal(realized_logs, expected_logs)


def test_normal_draw_logs_match_realized_values_at_200_samples():
    from increment.estimation._winsor_bootstrap import _realized_draw_and_logs

    latent_logs = np.linspace(-10.0, 10.0, 200)
    expected_draws = np.exp(latent_logs)
    expected_logs = np.log(expected_draws)
    with np.errstate(all="ignore"):
        draws, realized_logs = _realized_draw_and_logs(latent_logs)
    np.testing.assert_array_equal(draws, expected_draws)
    np.testing.assert_array_equal(realized_logs, expected_logs)


def test_realized_subnormal_logs_match_original_refit():
    from increment.estimation._winsor_bootstrap import (
        _realized_draw_and_logs,
        full_procedure_statistics_many,
    )

    latent_draw_logs = (
        np.asarray([[0.1, -710.0, 0.3, 0.5]]),
        np.asarray([[2.0, 2.5, 3.0, 3.5]]),
    )
    with np.errstate(all="ignore"):
        realized = tuple(_realized_draw_and_logs(logs) for logs in latent_draw_logs)
    samples = tuple(draw for draw, _ in realized)
    realized_logs = tuple(logs for _, logs in realized)
    assert 0 < samples[0][0, 1] < np.finfo(float).tiny
    assert realized_logs[0][0, 1] == np.log(samples[0][0, 1])
    old_refit = full_procedure_statistics_many(samples, 0.5, ((0, 1),))[0]
    cached_refit = full_procedure_statistics_many(
        samples, 0.5, ((0, 1),), log_samples=realized_logs
    )[0]
    np.testing.assert_array_equal(cached_refit.valid(), old_refit.valid())
    for field in ("log_relative", "additive", "log_se", "additive_se", "density_scaled"):
        np.testing.assert_allclose(
            getattr(cached_refit, field), getattr(old_refit, field), rtol=1e-12, atol=1e-12
        )


def test_density_scratch_preserves_kde_operation_results_exactly():
    from increment.estimation._winsor_bootstrap import _scaled_log_density

    log_cutoff = np.asarray([0.2, 0.7])
    logs = (
        np.asarray([[0.0, 0.1, 0.4], [0.5, 0.8, 1.1]]),
        np.asarray([[0.3, 0.6], [0.2, 0.9]]),
    )
    bandwidths = (np.asarray([0.2, 0.3]), np.asarray([0.25, 0.4]))
    total = sum(values.shape[1] for values in logs)
    expected = sum(
        np.sum(np.exp(-0.5 * ((log_cutoff[:, None] - values) / h[:, None]) ** 2), axis=1)
        / h
        / math.sqrt(2 * math.pi)
        / total
        for values, h in zip(logs, bandwidths, strict=True)
    )
    np.testing.assert_array_equal(
        _scaled_log_density(log_cutoff, logs, bandwidths, total), expected
    )


def test_gaussian_density_keeps_nonzero_subnormal_terms():
    from increment.estimation._winsor_bootstrap import _scaled_log_density

    log_cutoff = np.asarray([0.0, 0.0, 0.0])
    logs = (
        np.asarray(
            [
                [0.0, 3.85, 3.86, 3.87, -3.9],
                [3.85, 3.86, 3.87, -3.9, 3.9],
                [3.85, 3.86, 3.87, -3.9, 3.9],
            ]
        ),
    )
    bandwidths = (np.asarray([0.1, 0.1, 0.1]),)
    actual = _scaled_log_density(log_cutoff, logs, bandwidths, 5)
    expected = np.asarray(
        [
            np.sum(np.exp(-0.5 * ((log_cutoff[i] - logs[0][i]) / bandwidths[0][i]) ** 2))
            / bandwidths[0][i]
            / math.sqrt(2 * math.pi)
            / 5
            for i in range(3)
        ]
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual[1] > 0
    assert actual[2] > 0


def test_statistics_many_reuses_caller_owned_density_scratch():
    from increment.estimation._winsor_bootstrap import full_procedure_statistics_many

    samples = (
        np.asarray([[1.0, 2.0, 4.0], [1.0, 8.0, 12.0]]),
        np.asarray([[2.0, 3.0, 8.0, 10.0], [2.0, 3.0, 4.0, 5.0]]),
    )
    contrasts = ((0, 1),)
    expected = full_procedure_statistics_many(samples, 0.75, contrasts)[0]
    density_workspaces = tuple(np.empty_like(values) for values in samples)
    density_output = np.empty(2)
    actual = full_procedure_statistics_many(
        samples,
        0.75,
        contrasts,
        density_workspaces=density_workspaces,
        density_output=density_output,
    )[0]
    assert actual.density_scaled is density_output
    np.testing.assert_array_equal(actual.density_scaled, expected.density_scaled)


def test_row_bandwidth_standard_deviation_matches_numpy_exactly():
    from increment.estimation._winsor_bootstrap import _row_sample_std

    values = np.asarray(
        [
            [0.13, 0.22, -1.8, 0.0, 4.1, 2.7],
            [3.0, -1.0, 5.1, 2.4, -2.2, 0.0],
        ]
    )
    workspace = np.empty_like(values)
    actual = _row_sample_std(values, workspace)
    np.testing.assert_array_equal(actual, values.std(axis=1, ddof=1))


def test_reused_score_moments_match_explicit_studentization_variance():
    from increment.estimation._winsor_bootstrap import _row_score_moments

    rng = np.random.default_rng(310)
    outcomes = rng.lognormal(size=(4, 129))
    clipped = np.minimum(outcomes, 1.0)
    indicators = outcomes <= 1.0
    means = clipped.mean(axis=1)
    cdfs = indicators.mean(axis=1)
    coefficient = np.asarray([-1.4, 0.8, -0.25, 1.1])
    density_term = np.asarray([0.12, -0.35, 0.7, 0.04])

    variance, covariance, scales = _row_score_moments(clipped, indicators, cdfs)
    n = clipped.shape[1]
    indicator_variance = cdfs * (1.0 - cdfs) * n / (n - 1)
    scaled_coefficient = coefficient * scales
    expected = (
        scaled_coefficient**2 * variance
        + density_term**2 * indicator_variance
        - 2 * scaled_coefficient * density_term * covariance
    )
    scores = coefficient[:, None] * (clipped - means[:, None]) + density_term[:, None] * (
        cdfs[:, None] - indicators
    )
    np.testing.assert_allclose(expected, scores.var(axis=1, ddof=1), rtol=1e-12, atol=1e-12)


def test_exact_point_summary_matches_legacy_fraction_identity():
    from fractions import Fraction

    from increment.winsor import RawArm, WinsorRawState

    tiny = math.nextafter(0.0, 1.0)
    extremes = (-0.0, tiny, 1e-300, 1.0, 1.0000000000000002, 1e100, 1e300, 1.7976931348623157e308)
    raw = WinsorRawState(
        metric="m",
        study_id="exact-point-summary",
        missingness="error",
        quantile=0.73,
        arms=(
            RawArm(group_id="C", values=tuple(extremes[i % len(extremes)] for i in range(1024))),
            RawArm(
                group_id="T",
                values=tuple(extremes[(3 * i + 1) % len(extremes)] for i in range(1024)),
            ),
        ),
    )
    expected_values = sorted(value for arm in raw.arms for value in arm.values)
    rank = (len(expected_values) - 1) * raw.quantile
    left, right = math.floor(rank), math.ceil(rank)
    weight = Fraction(rank - left)
    expected_cutoff = float(
        (1 - weight) * Fraction(expected_values[left]) + weight * Fraction(expected_values[right])
    )
    expected_means = tuple(
        (
            arm.group_id,
            sum((Fraction(min(y, expected_cutoff)) for y in arm.values), Fraction())
            / len(arm.values),
        )
        for arm in raw.arms
    )
    summary = raw._exact_point_summary()
    assert summary == (expected_cutoff, expected_means)
    assert raw._exact_point_summary() == summary

    from increment.errors import CodedError
    from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_reference
    from increment.winsor import BootstrapReference

    reference = full_procedure_bootstrap_reference(_state(), "C", "T")
    payload = reference.model_dump()
    payload["raw"]["arms"][0]["values"] = (-0.0, 2.0, 3.0, 9.0)
    with pytest.raises(CodedError) as error:
        BootstrapReference.model_validate(payload)
    assert error.value.code == "estimation.winsor.pool_mismatch"

    invalid_centers = reference.model_dump()
    first_pilot = invalid_centers["pilots"][0]
    first_pilot["log_centers"] = tuple(reversed(first_pilot["log_centers"]))
    with pytest.raises(CodedError) as error:
        BootstrapReference.model_validate(invalid_centers)
    assert error.value.code == "estimation.winsor.pool_mismatch"

    copied = raw.model_copy(
        update={
            "arms": (
                RawArm(group_id="C", values=(1.0, 2.0)),
                RawArm(group_id="T", values=(3.0, 4.0)),
            )
        }
    )
    assert copied._exact_point_summary() != summary


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
        ((0, 1), "pilot_degenerate"),
        ((0, 0, 0), "pilot_degenerate"),
        ((-1, 1), "pilot_negative_outcome"),
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
