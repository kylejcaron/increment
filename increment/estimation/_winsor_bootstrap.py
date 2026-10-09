"""Positive log-kernel, arm-stratified full-procedure bootstrap-t.

All inner computations consume captured arrays. Density evaluation is O(B*N),
at the replicate cutoff only; no pairwise kernel matrix or warehouse access.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from itertools import starmap
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

from increment.winsor import (
    BootstrapReference,
    BootstrapRoots,
    PositiveLogPilot,
    WinsorConfidenceSet,
    WinsorRawState,
    _reference_context,
    bootstrap_point,
    bootstrap_root_interval,
    positive_log_bandwidth,
    raw_arm_zero_counts,
    refuse_cutoff_in_zero_atom,
    winsor_refuse,
)

_CHUNK = 64

_WORKING_SET_ELEMENTS = 1 << 20

_THREAD_POOL_MIN_BLOCK_ELEMENTS = 16_384


@dataclass(frozen=True, slots=True)
class _PilotData:
    """Read-only log centres kept as arrays until their portable tuples are needed.

    ``atom_centers`` prefixes ``zero_count`` entries of ``-inf`` to the log
    centres so that a resampled zero realises as exactly ``0.0`` and drops out
    of every log-kernel sum; a positive-only arm samples ``log_centers`` itself.
    """

    group_id: str
    log_centers: np.ndarray
    bandwidth: float
    zero_count: int = 0
    atom_centers: np.ndarray | None = None

    @property
    def size(self) -> int:
        return self.zero_count + len(self.log_centers)

    @property
    def sampling_centers(self) -> np.ndarray:
        return self.log_centers if self.atom_centers is None else self.atom_centers


def _block_rows(total_units: int) -> int:
    rows = min(_CHUNK, max(1, _WORKING_SET_ELEMENTS // total_units))
    return 1 << (rows.bit_length() - 1)


def _fit_positive_log_pilot_data(raw: WinsorRawState) -> tuple[_PilotData, ...]:
    pilots = []
    for arm, zero_count in zip(raw.arms, raw_arm_zero_counts(raw), strict=True):
        positives = arm.values[zero_count:]
        logs = np.fromiter((math.log(y) for y in positives), dtype=np.float64, count=len(positives))
        h = positive_log_bandwidth(logs)
        if not math.isfinite(h) or h <= 0:
            winsor_refuse("pilot_degenerate", "Each arm must have nonzero finite log variance.")
        logs.flags.writeable = False
        atom_centers = None
        if zero_count:
            atom_centers = np.concatenate([np.full(zero_count, -np.inf), logs])
            atom_centers.flags.writeable = False
        pilots.append(
            _PilotData(
                group_id=arm.group_id,
                log_centers=logs,
                bandwidth=h,
                zero_count=zero_count,
                atom_centers=atom_centers,
            )
        )
    return tuple(pilots)


def fit_positive_log_pilot(raw: WinsorRawState) -> tuple[PositiveLogPilot, ...]:
    return tuple(
        PositiveLogPilot(
            group_id=pilot.group_id,
            log_centers=tuple(float(center) for center in pilot.log_centers),
            bandwidth=pilot.bandwidth,
            zero_count=pilot.zero_count,
        )
        for pilot in _fit_positive_log_pilot_data(raw)
    )


def _materialize_pilot_data(pilot_data: list[_PilotData]) -> tuple[PositiveLogPilot, ...]:
    """Build portable pilot models after releasing the resampling arrays."""
    pilots = []
    while pilot_data:
        pilot = pilot_data.pop()
        pilots.append(
            PositiveLogPilot(
                group_id=pilot.group_id,
                log_centers=tuple(float(center) for center in pilot.log_centers),
                bandwidth=pilot.bandwidth,
                zero_count=pilot.zero_count,
            )
        )
        del pilot
    pilots.reverse()
    return tuple(pilots)


def pilot_parts(
    pilot: PositiveLogPilot | _PilotData, cutoff: float, *, center_array: np.ndarray | None = None
) -> tuple[float, float, float, float]:
    """CDF, density and two truncated moments of the hurdle lognormal mixture.

    Every quantity is averaged over the full arm size, so the atom contributes
    ``zero_count / n`` to the CDF and nothing to the density or the moments.
    """
    z, h = (
        (np.asarray(pilot.log_centers) if center_array is None else center_array),
        pilot.bandwidth,
    )
    n = pilot.zero_count + len(z)
    lc = math.log(cutoff)
    u = (lc - z) / h
    cdf = (pilot.zero_count + float(np.sum(ndtr(u)))) / n
    density = float(np.sum(np.exp(-u * u / 2) / math.sqrt(2 * math.pi))) / n / h / cutoff
    moments = []
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for k in (1, 2):
            log_moment = np.logaddexp(
                k * z + k * k * h * h / 2 + log_ndtr(u - k * h), k * lc + log_ndtr(-u)
            )
            # Divide before summing to avoid overflowing a representable mean.
            moments.append(float(np.sum(np.exp(log_moment - math.log(n)))))
    return cdf, density, moments[0], moments[1]


def pilot_population_target(
    pilots: Sequence[PositiveLogPilot | _PilotData],
    quantile: float,
    control: str,
    treatment: str,
    *,
    center_arrays: tuple[np.ndarray, ...] | None = None,
) -> tuple[float, float, float]:
    if center_arrays is None:
        center_arrays = tuple(np.asarray(p.log_centers) for p in pilots)
    total = sum(p.zero_count + len(p.log_centers) for p in pilots)
    zero_share = Fraction(sum(p.zero_count for p in pilots), total)
    if Fraction(quantile) <= zero_share:
        winsor_refuse(
            "cutoff_in_zero_atom",
            "The pilot population cutoff lies in the zero atom; raise upper_percentile above "
            "the pooled zero share or use a fixed upper_value.",
        )

    def objective(log_c):
        return (
            math.fsum(
                (pilot.zero_count + float(np.sum(ndtr((log_c - centers) / pilot.bandwidth))))
                / total
                for pilot, centers in zip(pilots, center_arrays, strict=True)
            )
            - quantile
        )

    low = min(min(p.log_centers) - 40 * p.bandwidth for p in pilots)
    high = max(max(p.log_centers) + 40 * p.bandwidth for p in pilots)
    log_c = brentq(objective, low, high, xtol=1e-13, rtol=1e-14)
    try:
        cutoff = math.exp(log_c)
    except OverflowError:
        winsor_refuse("endpoint_unrepresentable", "Pilot population cutoff exceeds numeric range.")
    if not math.isfinite(cutoff) or cutoff <= 0:
        winsor_refuse("endpoint_unrepresentable", "Pilot population cutoff is unrepresentable.")
    means = {
        pilot.group_id: pilot_parts(pilot, cutoff, center_array=centers)[2]
        for pilot, centers in zip(pilots, center_arrays, strict=True)
    }
    if any(not math.isfinite(x) or x <= 0 for x in means.values()):
        winsor_refuse("endpoint_unrepresentable", "Pilot population mean is unrepresentable.")
    return (
        cutoff,
        math.log(means[treatment]) - math.log(means[control]),
        means[treatment] - means[control],
    )


@dataclass(frozen=True)
class _Statistics:
    cutoff: np.ndarray
    log_relative: np.ndarray
    additive: np.ndarray
    log_se: np.ndarray
    additive_se: np.ndarray
    density_scaled: np.ndarray
    in_atom: np.ndarray | None = None

    def valid(self):
        usable = (
            np.isfinite(self.log_relative)
            & np.isfinite(self.additive)
            & np.isfinite(self.log_se)
            & (self.log_se > 0)
            & np.isfinite(self.additive_se)
            & (self.additive_se > 0)
            & np.isfinite(self.density_scaled)
            & (self.density_scaled > 0)
        )
        return usable if self.in_atom is None else usable & ~self.in_atom


def _row_sample_std(values: np.ndarray, work: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    np.subtract(values, mean, out=work)
    np.square(work, out=work)
    variance = np.sum(work, axis=1)
    variance /= values.shape[1] - 1
    np.sqrt(variance, out=variance)
    return variance


def _positive_row_sample_std(
    values: np.ndarray, positive_counts: np.ndarray, work: np.ndarray | None
) -> np.ndarray:
    """Row sample standard deviation over the positive entries only.

    Resampled zeros carry ``-inf`` logs; exactly those are replaced by zero
    before each reduction, so an overflowing positive draw (``+inf``) still
    poisons the row the way it does on a positive-only arm.
    """
    if work is None:
        work = np.empty_like(values)
    zero = np.isneginf(values)
    np.copyto(work, values)
    np.copyto(work, 0.0, where=zero)
    mean = np.sum(work, axis=1) / positive_counts
    np.subtract(values, mean[:, None], out=work)
    np.copyto(work, 0.0, where=zero)
    np.square(work, out=work)
    variance = np.sum(work, axis=1)
    variance /= positive_counts - 1
    np.sqrt(variance, out=variance)
    return variance


def _row_score_moments(
    clipped: np.ndarray,
    indicators: np.ndarray,
    cdfs: np.ndarray,
    work: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized value moments and their row scales."""
    n = clipped.shape[1]
    centered = clipped - clipped[:, :1] if work is None else work
    if work is not None:
        np.subtract(clipped, clipped[:, :1], out=centered)
    scales = np.max(np.abs(centered), axis=1)
    denominator = np.where(scales > 0, scales, 1.0)
    centered /= denominator[:, None]
    variance = centered.var(axis=1, ddof=1)
    centered_mean = centered.mean(axis=1)
    covariance = (np.sum(centered, axis=1, where=indicators) - n * centered_mean * cdfs) / (n - 1)
    return variance, covariance, scales


def _tail_type7_cutoff(
    samples: tuple[np.ndarray, ...], quantile: float, lower_bound: float
) -> np.ndarray:
    """Compute upper type-7 cutoffs from just the needed upper-tail order statistics."""
    total = sum(values.shape[1] for values in samples)
    if quantile < 0.9:
        return np.quantile(np.concatenate(samples, axis=1), quantile, axis=1, method="linear")

    rank = (total - 1) * quantile
    lower_rank = math.floor(rank)
    upper_rank = min(lower_rank + 1, total - 1)
    fraction = rank - lower_rank
    cutoffs = np.empty(samples[0].shape[0])
    for row_index in range(cutoffs.size):
        row_parts = tuple(values[row_index] for values in samples)
        if any(not np.isfinite(np.max(part)) for part in row_parts):
            cutoffs[row_index] = np.quantile(np.concatenate(row_parts), quantile, method="linear")
            continue
        tail_parts = tuple(part[part > lower_bound] for part in row_parts)
        tail_count = sum(part.size for part in tail_parts)
        if tail_count < total - lower_rank:
            cutoffs[row_index] = np.quantile(np.concatenate(row_parts), quantile, method="linear")
            continue
        tail = np.concatenate(tail_parts)
        low_index = tail_count - (total - lower_rank)
        high_index = tail_count - (total - upper_rank)
        tail.partition((low_index, high_index))
        low = tail[low_index]
        high = tail[high_index]
        difference = high - low
        value = low + difference * fraction
        if fraction >= 0.5:
            value = high - difference * (1.0 - fraction)
        cutoffs[row_index] = value
    return cutoffs


def _map_arms[T](
    executor: Executor | None,
    operation: Callable[..., T],
    arguments: Sequence[tuple[Any, ...]],
) -> tuple[T, ...]:
    if executor is None:
        return tuple(starmap(operation, arguments))
    futures = [executor.submit(operation, *args) for args in arguments]
    return tuple(future.result() for future in futures)


def _scaled_log_density_arm(
    log_cutoff: np.ndarray,
    values: np.ndarray,
    bandwidth: np.ndarray,
    total: int,
    work: np.ndarray,
) -> np.ndarray:
    with np.errstate(all="ignore"):
        np.subtract(log_cutoff[:, None], values, out=work)
        np.divide(work, bandwidth[:, None], out=work)
        np.square(work, out=work)
        np.multiply(work, -0.5, out=work)
        np.exp(work, out=work)
        return np.sum(work, axis=1) / bandwidth / math.sqrt(2 * math.pi) / total


def _scaled_log_density(
    log_cutoff: np.ndarray,
    logs: tuple[np.ndarray, ...],
    bandwidths: tuple[np.ndarray, ...],
    total: int,
    *,
    workspaces: tuple[np.ndarray, ...] | None = None,
    density_output: np.ndarray | None = None,
    executor: Executor | None = None,
) -> np.ndarray:
    if density_output is None:
        density_scaled = np.zeros_like(log_cutoff)
    else:
        density_scaled = density_output
        density_scaled.fill(0)
    arguments = tuple(
        (
            log_cutoff,
            values,
            bandwidth,
            total,
            np.empty_like(values) if workspaces is None else workspaces[index],
        )
        for index, (values, bandwidth) in enumerate(zip(logs, bandwidths, strict=True))
    )
    density_parts = _map_arms(executor, _scaled_log_density_arm, arguments)
    for density_part in density_parts:
        density_scaled += density_part
    return density_scaled


def _realized_draw_and_logs(
    log_draw: np.ndarray, *, draw_output: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if draw_output is None:
        draw = np.exp(log_draw)
    else:
        np.exp(log_draw, out=draw_output)
        draw = draw_output
    np.log(draw, out=log_draw)
    return draw, log_draw


def full_procedure_statistics(
    samples: tuple[np.ndarray, ...],
    quantile: float,
    ci: int,
    ti: int,
    *,
    cutoff: np.ndarray | None = None,
    log_samples: tuple[np.ndarray, ...] | None = None,
) -> _Statistics:
    """Vectorized complete refit; rows are independent scheduled replicates."""
    return full_procedure_statistics_many(
        samples,
        quantile,
        ((ci, ti),),
        cutoff=cutoff,
        log_samples=log_samples,
    )[0]


def _score_arm_contribution(
    y: np.ndarray,
    clipped: np.ndarray,
    mean: np.ndarray,
    cdf: np.ndarray,
    indicator: np.ndarray,
    coefficient: np.ndarray,
    derivative: np.ndarray,
    density_scaled: np.ndarray,
    total: int,
    score_moment: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    quadratic_limit: float,
) -> np.ndarray:
    with np.errstate(all="ignore"):
        n = y.shape[1]
        b = derivative * (n / total) / density_scaled
        if score_moment is None:
            score = coefficient[:, None] * (clipped - mean[:, None]) + b[:, None] * (
                cdf[:, None] - indicator
            )
            return score.var(axis=1, ddof=1) / n

        clipped_variance, clipped_indicator_covariance, clipped_scale = score_moment
        indicator_variance = cdf * (1 - cdf) * n / (n - 1)
        value_scale = np.abs(coefficient) * clipped_scale
        scale = np.maximum(value_scale, np.abs(b))
        finite_scale = np.isfinite(scale)
        denominator = np.where(finite_scale & (scale > 0), scale, 1.0)
        scaled_a = np.where(finite_scale, coefficient / denominator, 0.0) * clipped_scale
        scaled_b = np.where(finite_scale, b / denominator, 0.0)
        finite_coefficients = np.isfinite(scaled_a) & np.isfinite(scaled_b)
        term_a = scaled_a * scaled_a * clipped_variance
        term_b = scaled_b * scaled_b * indicator_variance
        cross = 2 * scaled_a * scaled_b * clipped_indicator_covariance
        normalized_variance = term_a + term_b - cross
        magnitude = np.abs(term_a) + np.abs(term_b) + np.abs(cross)
        unstable = (
            ~finite_scale
            | ~finite_coefficients
            | ~np.isfinite(normalized_variance)
            | (normalized_variance < 0)
            | (scale > quadratic_limit)
            | ((magnitude > 0) & (normalized_variance <= 64 * np.finfo(float).eps * magnitude))
        )
        contribution = denominator * np.sqrt(np.maximum(normalized_variance, 0) / n)
        legacy_degenerate = (contribution > 0) & (contribution * contribution == 0)
        for row in np.flatnonzero(unstable | legacy_degenerate):
            score = coefficient[row] * (clipped[row] - mean[row]) + b[row] * (
                cdf[row] - indicator[row]
            )
            contribution[row] = np.sqrt(np.var(score, ddof=1) / n)
        return contribution


def _row_score_moments_safe(
    clipped: np.ndarray,
    indicators: np.ndarray,
    cdfs: np.ndarray,
    work: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        return _row_score_moments(clipped, indicators, cdfs, work)


def _arm_standard_deviation(
    values: np.ndarray, work: np.ndarray | None, zero_counts: np.ndarray | None = None
) -> np.ndarray:
    with np.errstate(all="ignore"):
        if zero_counts is not None:
            return _positive_row_sample_std(values, values.shape[1] - zero_counts, work)
        return values.std(axis=1, ddof=1) if work is None else _row_sample_std(values, work)


def _zero_draw_counts(
    samples: tuple[np.ndarray, ...], atoms: tuple[bool, ...] | None
) -> tuple[np.ndarray | None, ...]:
    """Per-arm zero counts per row for arms that carry an atom; None otherwise."""
    if atoms is None:
        return (None,) * len(samples)
    return tuple(
        np.count_nonzero(y == 0, axis=1) if atom else None
        for y, atom in zip(samples, atoms, strict=True)
    )


def full_procedure_statistics_many(
    samples: tuple[np.ndarray, ...],
    quantile: float,
    contrasts: tuple[tuple[int, int], ...],
    *,
    cutoff: np.ndarray | None = None,
    log_samples: tuple[np.ndarray, ...] | None = None,
    density_workspaces: tuple[np.ndarray, ...] | None = None,
    tail_bound: float | None = None,
    density_output: np.ndarray | None = None,
    _executor: Executor | None = None,
    atoms: tuple[bool, ...] | None = None,
) -> tuple[_Statistics, ...]:
    """Refit several contrasts from one complete all-arm pool evaluation.

    ``atoms`` marks the arms whose pilot carries a zero atom. Their zero draws
    realise as ``0.0`` with ``-inf`` logs: they count toward the pooled cutoff,
    the clipped means and the indicators, drop out of the log-kernel density
    and the bandwidth, and a replicate whose cutoff position falls inside the
    atom is not studentizable.
    """
    total = sum(y.shape[1] for y in samples)
    zero_counts = _zero_draw_counts(samples, atoms)
    in_atom: np.ndarray | None = None
    if any(count is not None for count in zero_counts):
        zero_rows = np.zeros(samples[0].shape[0], dtype=np.int64)
        for count in zero_counts:
            if count is not None:
                zero_rows += count
        in_atom = zero_rows > math.floor((total - 1) * quantile)
    with np.errstate(all="ignore"):
        if cutoff is None:
            if tail_bound is None:
                cutoff = np.quantile(
                    np.concatenate(samples, axis=1), quantile, axis=1, method="linear"
                )
            else:
                cutoff = _tail_type7_cutoff(samples, quantile, tail_bound)
        log_c = np.log(cutoff)
        logs = tuple(np.log(y) for y in samples) if log_samples is None else log_samples
        workspaces = (None,) * len(logs) if density_workspaces is None else density_workspaces
        standard_deviations = _map_arms(
            _executor,
            _arm_standard_deviation,
            tuple(zip(logs, workspaces, zero_counts, strict=True)),
        )
        bandwidths = tuple(
            1.06 * standard_deviation * z.shape[1] ** -0.2
            if count is None
            else 1.06 * standard_deviation * (z.shape[1] - count) ** -0.2
            for z, standard_deviation, count in zip(
                logs, standard_deviations, zero_counts, strict=True
            )
        )
        density_scaled = _scaled_log_density(
            log_c,
            logs,
            bandwidths,
            total,
            workspaces=density_workspaces,
            density_output=density_output,
            executor=_executor,
        )
        clipped = tuple(np.minimum(y / cutoff[:, None], 1) for y in samples)
        means = tuple(w.mean(axis=1) for w in clipped)
        indicators = tuple(y <= cutoff[:, None] for y in samples)
        cdfs = tuple(i.mean(axis=1) for i in indicators)
        if total > 4_000:
            scratch = density_workspaces or (None,) * len(samples)
            score_moments = _map_arms(
                _executor,
                _row_score_moments_safe,
                tuple(
                    (w, indicator, f, work)
                    for w, indicator, f, work in zip(
                        clipped, indicators, cdfs, scratch, strict=True
                    )
                ),
            )
        else:
            score_moments = None
        results = []
        for ci, ti in contrasts:
            ses = []
            for relative in (True, False):
                coefficients = [np.zeros_like(cutoff) for _ in samples]
                coefficients[ci] = -1 / means[ci] if relative else -np.ones_like(cutoff)
                coefficients[ti] = 1 / means[ti] if relative else np.ones_like(cutoff)
                derivative = sum(a * (1 - f) for a, f in zip(coefficients, cdfs, strict=True))
                quadratic_limit = math.sqrt(np.finfo(float).max / len(samples))
                contributions = _map_arms(
                    _executor,
                    _score_arm_contribution,
                    tuple(
                        (
                            y,
                            w,
                            m,
                            f,
                            indicator,
                            coefficient,
                            derivative,
                            density_scaled,
                            total,
                            None if score_moments is None else score_moments[arm_index],
                            quadratic_limit,
                        )
                        for arm_index, (y, w, m, f, indicator, coefficient) in enumerate(
                            zip(
                                samples, clipped, means, cdfs, indicators, coefficients, strict=True
                            )
                        )
                    ),
                )
                if score_moments is None:
                    variance = np.zeros_like(cutoff)
                    for contribution in contributions:
                        variance += contribution
                    se = np.sqrt(variance)
                else:
                    se = np.zeros_like(cutoff)
                    for contribution in contributions:
                        se = np.hypot(se, contribution)
                ses.append(se if relative else se * cutoff)
            results.append(
                _Statistics(
                    cutoff,
                    np.log(means[ti]) - np.log(means[ci]),
                    (means[ti] - means[ci]) * cutoff,
                    ses[0],
                    ses[1],
                    density_scaled,
                    in_atom,
                )
            )
        return tuple(results)


def _bootstrap_observed_points(
    control_mean: Fraction, treatment_means: tuple[Fraction, ...]
) -> tuple[tuple[float, float], ...]:
    points = []
    for mean in treatment_means:
        try:
            relative_point = float((mean - control_mean) / control_mean)
        except OverflowError:
            relative_point = math.inf
        ell = (
            math.log1p(relative_point)
            if math.isfinite(relative_point) and relative_point > -1
            else math.log(float(mean)) - math.log(float(control_mean))
        )
        points.append((ell, float(mean - control_mean)))
    return tuple(points)


def _append_pivot_block(
    log_roots: list[list[float | None]],
    additive_roots: list[list[float | None]],
    failures: list[list[int]],
    pivots: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    first_index: int,
) -> None:
    for treatment_index, (relative, additive, valid) in enumerate(pivots):
        usable = valid & np.isfinite(relative) & np.isfinite(additive)
        relative_values = relative.tolist()
        additive_values = additive.tolist()
        if not usable.all():
            for row in np.flatnonzero(~usable):
                relative_values[row] = None
                additive_values[row] = None
                failures[treatment_index].append(first_index + int(row))
        log_roots[treatment_index].extend(relative_values)
        additive_roots[treatment_index].extend(additive_values)


def _sample_bootstrap_arm(
    pilot: _PilotData,
    z: np.ndarray,
    generators: tuple[np.random.Generator, np.random.Generator],
    log_buffer: np.ndarray,
    draw_buffer: np.ndarray,
    noise_buffer: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers, noise = generators
    shape = (size, len(z))
    elements = size * len(z)
    log_draw = log_buffer[:elements].reshape(shape)
    draw_output = draw_buffer[:elements].reshape(shape)
    noise_draw = noise_buffer[:elements].reshape(shape)
    with np.errstate(all="ignore"):
        indices = centers.integers(len(z), size=shape)
        np.take(z, indices, out=log_draw)
        noise.standard_normal(size=shape, out=noise_draw)
        np.multiply(noise_draw, pilot.bandwidth, out=noise_draw)
        np.add(log_draw, noise_draw, out=log_draw)
        return _realized_draw_and_logs(log_draw, draw_output=draw_output)


def _sample_bootstrap_block(
    pilot_data: list[_PilotData],
    pilot_center_arrays: tuple[np.ndarray, ...],
    generators: tuple[tuple[np.random.Generator, np.random.Generator], ...],
    log_buffers: tuple[np.ndarray, ...],
    draw_buffers: tuple[np.ndarray, ...],
    noise_buffers: tuple[np.ndarray, ...],
    size: int,
    *,
    executor: Executor | None = None,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    samples = _map_arms(
        executor,
        _sample_bootstrap_arm,
        tuple(
            (pilot, z, arm_generators, log_buffer, draw_buffer, noise_buffer, size)
            for pilot, z, arm_generators, log_buffer, draw_buffer, noise_buffer in zip(
                pilot_data,
                pilot_center_arrays,
                generators,
                log_buffers,
                draw_buffers,
                noise_buffers,
                strict=True,
            )
        ),
    )
    draws, draw_logs = zip(*samples, strict=True)
    return tuple(draws), tuple(draw_logs)


def full_procedure_bootstrap_references(
    raw: WinsorRawState,
    control: str,
    treatments: tuple[str, ...],
    *,
    validation_context: Mapping[object, object] | None = None,
) -> dict[str, BootstrapReference]:
    """Build treatment references using one pooled draw and cutoff pass."""
    if (
        raw.executed_method != "positive-log-kernel-bootstrap-t-v1"
        or not treatments
        or len(set(treatments)) != len(treatments)
        or control in treatments
    ):
        winsor_refuse(
            "invalid_state", "Bootstrap requires its resolved method and distinct contrast arms."
        )
    raw.arm(control)
    for treatment in treatments:
        raw.arm(treatment)
    pilot_data = list(_fit_positive_log_pilot_data(raw))
    pilot_center_arrays = tuple(p.log_centers for p in pilot_data)
    sampling_arrays = tuple(p.sampling_centers for p in pilot_data)
    atoms = tuple(p.zero_count > 0 for p in pilot_data)
    groups = [p.group_id for p in pilot_data]
    ci = groups.index(control)
    contrast_indices = tuple((ci, groups.index(treatment)) for treatment in treatments)
    point_summary, reference_context = _reference_context(raw, validation_context)
    cutoff, exact_means = point_summary
    means_by_group = dict(exact_means)
    refuse_cutoff_in_zero_atom(raw, sum(p.zero_count for p in pilot_data))
    samples = tuple(np.asarray(a.values)[None, :] for a in raw.arms)
    observed_statistics = full_procedure_statistics_many(
        samples, raw.quantile, contrast_indices, cutoff=np.asarray([cutoff]), atoms=atoms
    )
    for observed in observed_statistics:
        if not observed.valid()[0]:
            reason = (
                "density_unresolved"
                if not math.isfinite(float(observed.density_scaled[0]))
                or observed.density_scaled[0] <= 0
                else "studentization_degenerate"
            )
            winsor_refuse(reason, "Observed full-procedure studentization is unavailable.")
    del samples
    spec = raw.inference
    pilot_targets = tuple(
        pilot_population_target(
            pilot_data, raw.quantile, control, treatment, center_arrays=pilot_center_arrays
        )
        for treatment in treatments
    )
    points = _bootstrap_observed_points(
        means_by_group[control],
        tuple(means_by_group[treatment] for treatment in treatments),
    )
    generators = tuple(
        (
            np.random.Generator(
                np.random.PCG64DXSM(
                    np.random.SeedSequence(spec.seed, spawn_key=(spec.stream, g, 0))
                )
            ),
            np.random.Generator(
                np.random.PCG64DXSM(
                    np.random.SeedSequence(spec.seed, spawn_key=(spec.stream, g, 1))
                )
            ),
        )
        for g in range(len(pilot_data))
    )
    log_roots = [[] for _ in treatments]
    additive_roots = [[] for _ in treatments]
    failures = [[] for _ in treatments]
    total_units = sum(p.size for p in pilot_data)
    block_rows = _block_rows(total_units)
    large_pool = total_units > 4_000
    density_work_buffers = (
        tuple(np.empty((block_rows, p.size)) for p in pilot_data) if large_pool else None
    )
    density_output_buffer = np.empty(block_rows) if large_pool else None
    log_buffers = tuple(np.empty(block_rows * len(z)) for z in sampling_arrays)
    draw_buffers = tuple(np.empty_like(buffer) for buffer in log_buffers)
    noise_buffers = tuple(np.empty_like(buffer) for buffer in log_buffers)
    tail_bound = cutoff * 0.9 if large_pool and raw.quantile >= 0.9 else None
    executor_context = (
        ThreadPoolExecutor(max_workers=len(pilot_data))
        if len(pilot_data) > 1 and total_units * block_rows >= _THREAD_POOL_MIN_BLOCK_ELEMENTS
        else nullcontext(None)
    )
    with executor_context as executor:
        for start in range(0, spec.replicates, _CHUNK):
            logical_size = min(_CHUNK, spec.replicates - start)
            for offset in range(0, logical_size, block_rows):
                size = min(block_rows, logical_size - offset)
                with np.errstate(all="ignore"):
                    draws, draw_logs = _sample_bootstrap_block(
                        pilot_data,
                        sampling_arrays,
                        generators,
                        log_buffers,
                        draw_buffers,
                        noise_buffers,
                        size,
                        executor=executor,
                    )
                    statistics = full_procedure_statistics_many(
                        draws,
                        raw.quantile,
                        contrast_indices,
                        log_samples=draw_logs,
                        tail_bound=tail_bound,
                        density_workspaces=(
                            None
                            if density_work_buffers is None
                            else tuple(work[:size] for work in density_work_buffers)
                        ),
                        density_output=(
                            None if density_output_buffer is None else density_output_buffer[:size]
                        ),
                        _executor=executor,
                        atoms=atoms,
                    )
                    pivots = []
                    for statistic, (center_log, center_delta) in zip(
                        statistics,
                        ((target[1], target[2]) for target in pilot_targets),
                        strict=True,
                    ):
                        pivots.append(
                            (
                                (statistic.log_relative - center_log) / statistic.log_se,
                                (statistic.additive - center_delta) / statistic.additive_se,
                                statistic.valid(),
                            )
                        )
                _append_pivot_block(
                    log_roots,
                    additive_roots,
                    failures,
                    pivots,
                    first_index=start + offset,
                )
    del (
        draws,
        draw_logs,
        statistics,
        pivots,
        density_work_buffers,
        density_output_buffer,
        log_buffers,
        draw_buffers,
        noise_buffers,
        pilot_center_arrays,
        sampling_arrays,
    )
    pilots = _materialize_pilot_data(pilot_data)

    references = {}
    for i, treatment in enumerate(treatments):
        observed = observed_statistics[i]
        center_cutoff, center_log, center_delta = pilot_targets[i]
        ell, delta = points[i]
        references[treatment] = BootstrapReference.model_validate(
            {
                "raw": raw,
                "spec": spec,
                "control": control,
                "treatment": treatment,
                "pilots": pilots,
                "pilot_cutoff": center_cutoff,
                "observed_cutoff": cutoff,
                "log_relative": BootstrapRoots(
                    point=ell,
                    pilot_target=center_log,
                    se=float(observed.log_se[0]),
                    roots=tuple(log_roots[i]),
                ),
                "additive": BootstrapRoots(
                    point=delta,
                    pilot_target=center_delta,
                    se=float(observed.additive_se[0]),
                    roots=tuple(additive_roots[i]),
                ),
                "failure_indices": tuple(failures[i]),
            },
            context=reference_context,
        )
    return references


def full_procedure_bootstrap_reference(
    raw: WinsorRawState, control: str, treatment: str
) -> BootstrapReference:
    return full_procedure_bootstrap_references(raw, control, (treatment,))[treatment]


def bootstrap_confidence_set(
    reference: BootstrapReference,
    alpha: float = 0.05,
    *,
    validation_context: Mapping[object, object] | None = None,
) -> WinsorConfidenceSet:
    """Invert the stored roots at ``alpha``; the context carries the exact point summary."""
    if not 0 < alpha < 1:
        winsor_refuse("invalid_state", "alpha must be strictly between zero and one.")
    _, validation_context = _reference_context(reference.raw, validation_context)
    return WinsorConfidenceSet.model_validate(
        {
            "raw": reference.raw,
            "control": reference.control,
            "treatment": reference.treatment,
            "alpha": alpha,
            "reference": reference,
            "relative": bootstrap_root_interval(reference, alpha, relative=True),
            "additive": bootstrap_root_interval(reference, alpha, relative=False),
            "point": bootstrap_point(reference),
            "additive_point": reference.additive.point,
        },
        context=validation_context,
    )


def bootstrap_p_value(reference: BootstrapReference, null: float, *, relative: bool) -> float:
    """Invert the stored equal-tail effect test; this is not a permutation p-value."""
    if reference.failure_indices:
        return 1.0
    if relative and null <= -1:
        return 0.0
    series = reference.log_relative if relative else reference.additive
    pivot = (series.point - (math.log1p(null) if relative else null)) / series.se
    roots = tuple(x for x in series.roots if x is not None)
    # Strict endpoint rejection with conservative equality and finite B resolution.
    tail = min(sum(x <= pivot for x in roots), sum(x >= pivot for x in roots))
    return min(1.0, 2 * (tail + 1) / (len(roots) + 1))
