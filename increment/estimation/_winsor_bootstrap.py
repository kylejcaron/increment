"""Positive log-kernel, arm-stratified full-procedure bootstrap-t.

All inner computations consume captured arrays. Density evaluation is O(B*N),
at the replicate cutoff only; no pairwise kernel matrix or warehouse access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

from increment.winsor import (
    BootstrapReference,
    BootstrapRoots,
    PositiveLogPilot,
    WinsorConfidenceSet,
    WinsorRawState,
    bootstrap_point,
    bootstrap_root_interval,
    positive_log_bandwidth,
    winsor_refuse,
)

_CHUNK = 64


def fit_positive_log_pilot(raw: WinsorRawState) -> tuple[PositiveLogPilot, ...]:
    pilots = []
    for arm in raw.arms:
        if arm.values[0] <= 0:
            winsor_refuse(
                "pilot_nonpositive_outcome",
                "Log-kernel inference requires strictly positive raw outcomes.",
            )
        if len(arm.values) < 2:
            winsor_refuse(
                "pilot_degenerate", "Log-kernel fitting requires at least two observations per arm."
            )
        logs = tuple(math.log(y) for y in arm.values)
        h = positive_log_bandwidth(logs)
        if not math.isfinite(h) or h <= 0:
            winsor_refuse("pilot_degenerate", "Each arm must have nonzero finite log variance.")
        pilots.append(PositiveLogPilot(group_id=arm.group_id, log_centers=logs, bandwidth=h))
    return tuple(pilots)


def pilot_parts(pilot: PositiveLogPilot, cutoff: float) -> tuple[float, float, float, float]:
    """CDF, density and two truncated moments of the explicit lognormal mixture."""
    z, h = np.asarray(pilot.log_centers), pilot.bandwidth
    lc = math.log(cutoff)
    u = (lc - z) / h
    cdf = float(np.mean(ndtr(u)))
    density = float(np.mean(np.exp(-u * u / 2) / math.sqrt(2 * math.pi))) / h / cutoff
    moments = []
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for k in (1, 2):
            log_moment = np.logaddexp(
                k * z + k * k * h * h / 2 + log_ndtr(u - k * h), k * lc + log_ndtr(-u)
            )
            # Divide before summing to avoid overflowing a representable mean.
            moments.append(float(np.sum(np.exp(log_moment - math.log(len(z))))))
    return cdf, density, moments[0], moments[1]


def pilot_population_target(
    pilots: tuple[PositiveLogPilot, ...], quantile: float, control: str, treatment: str
) -> tuple[float, float, float]:
    total = sum(len(p.log_centers) for p in pilots)

    def objective(log_c):
        return (
            math.fsum(
                float(np.sum(ndtr((log_c - np.asarray(p.log_centers)) / p.bandwidth))) / total
                for p in pilots
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
    means = {p.group_id: pilot_parts(p, cutoff)[2] for p in pilots}
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

    def valid(self):
        return (
            np.isfinite(self.log_relative)
            & np.isfinite(self.additive)
            & np.isfinite(self.log_se)
            & (self.log_se > 0)
            & np.isfinite(self.additive_se)
            & (self.additive_se > 0)
            & np.isfinite(self.density_scaled)
            & (self.density_scaled > 0)
        )


def full_procedure_statistics(
    samples: tuple[np.ndarray, ...],
    quantile: float,
    ci: int,
    ti: int,
    *,
    cutoff: np.ndarray | None = None,
) -> _Statistics:
    """Vectorized complete refit; rows are independent scheduled replicates."""
    total = sum(y.shape[1] for y in samples)
    with np.errstate(all="ignore"):
        if cutoff is None:
            cutoff = np.quantile(np.concatenate(samples, axis=1), quantile, axis=1, method="linear")
        log_c = np.log(cutoff)
        logs = tuple(np.log(y) for y in samples)
        bandwidths = tuple(1.06 * z.std(axis=1, ddof=1) * z.shape[1] ** -0.2 for z in logs)
        density_scaled = sum(
            np.sum(np.exp(-0.5 * ((log_c[:, None] - z) / h[:, None]) ** 2), axis=1)
            / h
            / math.sqrt(2 * math.pi)
            / total
            for z, h in zip(logs, bandwidths, strict=True)
        )
        clipped = tuple(np.minimum(y / cutoff[:, None], 1) for y in samples)
        means = tuple(w.mean(axis=1) for w in clipped)
        indicators = tuple(y <= cutoff[:, None] for y in samples)
        cdfs = tuple(i.mean(axis=1) for i in indicators)
        ses = []
        for relative in (True, False):
            coefficients = [np.zeros_like(cutoff) for _ in samples]
            coefficients[ci] = -1 / means[ci] if relative else -np.ones_like(cutoff)
            coefficients[ti] = 1 / means[ti] if relative else np.ones_like(cutoff)
            derivative = sum(a * (1 - f) for a, f in zip(coefficients, cdfs, strict=True))
            variance = np.zeros_like(cutoff)
            for y, w, m, f, indicator, a in zip(
                samples, clipped, means, cdfs, indicators, coefficients, strict=True
            ):
                n = y.shape[1]
                b = derivative * (n / total) / density_scaled
                score = a[:, None] * (w - m[:, None]) + b[:, None] * (f[:, None] - indicator)
                variance += score.var(axis=1, ddof=1) / n
            se = np.sqrt(variance)
            ses.append(se if relative else se * cutoff)
        return _Statistics(
            cutoff,
            np.log(means[ti]) - np.log(means[ci]),
            (means[ti] - means[ci]) * cutoff,
            ses[0],
            ses[1],
            density_scaled,
        )


def full_procedure_bootstrap_reference(
    raw: WinsorRawState, control: str, treatment: str
) -> BootstrapReference:
    from increment.estimation.winsor import _linear_cutoff, _mean_fraction

    if raw.inference.method != "positive-log-kernel-bootstrap-t-v1" or control == treatment:
        winsor_refuse(
            "invalid_state", "Bootstrap requires its explicit method and distinct contrast arms."
        )
    raw.arm(control)
    raw.arm(treatment)
    pilots = fit_positive_log_pilot(raw)
    groups = [p.group_id for p in pilots]
    ci, ti = groups.index(control), groups.index(treatment)
    samples = tuple(np.asarray(a.values)[None, :] for a in raw.arms)
    cutoff = _linear_cutoff(raw)
    observed = full_procedure_statistics(samples, raw.quantile, ci, ti, cutoff=np.asarray([cutoff]))
    if not observed.valid()[0]:
        reason = (
            "density_unresolved"
            if not math.isfinite(float(observed.density_scaled[0]))
            or observed.density_scaled[0] <= 0
            else "studentization_degenerate"
        )
        winsor_refuse(reason, "Observed full-procedure studentization is unavailable.")
    # Keep the exact type-7 clipped-arm statistic, independent of score scaling.
    mc = _mean_fraction(np.minimum(raw.arm(control).values, cutoff))
    mt = _mean_fraction(np.minimum(raw.arm(treatment).values, cutoff))
    try:
        relative_point = float((mt - mc) / mc)
    except OverflowError:
        relative_point = math.inf
    ell = (
        math.log1p(relative_point)
        if math.isfinite(relative_point) and relative_point > -1
        else math.log(float(mt)) - math.log(float(mc))
    )
    delta = float(mt - mc)
    pilot_cutoff, center_log, center_delta = pilot_population_target(
        pilots, raw.quantile, control, treatment
    )
    spec = raw.inference
    # Separate center/noise streams per canonical arm. Chunk size is part of v1.
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
        for g in range(len(pilots))
    )
    log_roots: list[float | None] = []
    additive_roots: list[float | None] = []
    failures = []
    for start in range(0, spec.replicates, _CHUNK):
        size = min(_CHUNK, spec.replicates - start)
        draws = []
        with np.errstate(all="ignore"):
            for pilot, (centers, noise) in zip(pilots, generators, strict=True):
                z = np.asarray(pilot.log_centers)
                indices = centers.integers(len(z), size=(size, len(z)))
                draws.append(
                    np.exp(z[indices] + pilot.bandwidth * noise.standard_normal((size, len(z))))
                )
            statistics = full_procedure_statistics(tuple(draws), raw.quantile, ci, ti)
            lr = (statistics.log_relative - center_log) / statistics.log_se
            ar = (statistics.additive - center_delta) / statistics.additive_se
        valid = statistics.valid() & np.isfinite(lr) & np.isfinite(ar)
        for i in range(size):
            log_roots.append(float(lr[i]) if valid[i] else None)
            additive_roots.append(float(ar[i]) if valid[i] else None)
            if not valid[i]:
                failures.append(start + i)
    return BootstrapReference(
        raw=raw,
        spec=spec,
        control=control,
        treatment=treatment,
        pilots=pilots,
        pilot_cutoff=pilot_cutoff,
        observed_cutoff=cutoff,
        log_relative=BootstrapRoots(
            point=ell, pilot_target=center_log, se=float(observed.log_se[0]), roots=tuple(log_roots)
        ),
        additive=BootstrapRoots(
            point=delta,
            pilot_target=center_delta,
            se=float(observed.additive_se[0]),
            roots=tuple(additive_roots),
        ),
        failure_indices=tuple(failures),
    )


def bootstrap_confidence_set(
    reference: BootstrapReference, alpha: float = 0.05
) -> WinsorConfidenceSet:
    if not 0 < alpha < 1:
        winsor_refuse("invalid_state", "alpha must be strictly between zero and one.")
    return WinsorConfidenceSet(
        raw=reference.raw,
        control=reference.control,
        treatment=reference.treatment,
        alpha=alpha,
        reference=reference,
        relative=bootstrap_root_interval(reference, alpha, relative=True),
        additive=bootstrap_root_interval(reference, alpha, relative=False),
        point=bootstrap_point(reference),
        additive_point=reference.additive.point,
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
