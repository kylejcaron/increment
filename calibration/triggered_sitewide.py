"""Certify calendar-aligned sitewide intervals against realized causal totals.

For iid two-arm sums, I_hat = N_selected * delta_hat and
V0_hat = V - n_t_selected * delta_hat. Their standard errors are N_selected
* se_delta and abs(N_selected * V / V0_hat**2) * se_delta. Ratio components
are adjusted separately: N0 = Vnum - nt * delta_num, D0 = Vden - nt *
delta_den, N1 = N0 + N_selected * delta_num, D1 = D0 + N_selected *
delta_den. Impact is N1/D1 - N0/D0, relative impact divides by N0/D0;
the quotient-rule gradients contract the full numerator/denominator
covariance of the independent arm means. Arm variances are Bessel-corrected.
The iid production reference is Normal.isf(alpha/2), not a t reference.

The domain requires at least two units in each selected arm, positive sum
baseline, and positive ratio denominator means, D0, D1 and N0. The delta-method
interval conditions on the realized site volume and membership; it omits
future-traffic and estimated-reach uncertainty. Coverage targets are the
realized finite-population I* = sum(tau_enrolled), V0* = V - sum(tau_treated)
and rel* = I*/V0*, not a superpopulation mean. Ratio truths apply the same
potential-outcome accounting to both components. Never-triggered and
non-enrolled outcomes remain in V even for the triggered reading.

T3 retains the C.2 outcomes and adds a fixed unaffected site background of
1,000,000 solely to admit a positive baseline for the broad zero-mean metric.
Only its absolute-impact variances are compared; these do not depend on V.

Sampling reduces the same per-arm centered additive moments used by ArmStats.
Inference is vectorized over replications to avoid repeated model validation;
the batched two-arm iid construction is tested against SitewideContrast /
SitewideRatioContrast and the production functions to absolute tolerance 1e-12
on randomized inputs. This equivalence is required before the full campaign.

The manifest below is frozen before sampling. Every design uses independent
replication seeds 1000 + rep; readings within a design are paired. Full gates
share the 67-statistic family allocation with other certification campaigns.
No rejected replication is replaced: failing to attain the frozen eligible
count refuses the region. An assigned failure stops the campaign and requires
escalation; a triggered failure requires refusing that triggered design region.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

from increment.estimation.armstats import ArmStats
from tests.mc import binomial_error_upper_bound, coverage_lower_bound, mcse

GATED_REPS = 107_100
ETA = 0.00007462686567164179
COVERAGE_FLOOR = 0.945
ALPHA = 0.05
SEED_START = 1000
SMOKE_REPS = 3
SENTINEL_REPS = 2000
SENTINEL_ETA = 0.01
SENTINEL_FLOOR = 0.925
VARIANCE_BACKGROUND = 1_000_000.0


@dataclass(frozen=True)
class Design:
    id: str
    kind: Literal["sum", "ratio", "variance", "composition"]
    reps: int
    n_site: int
    n_enrolled: int
    p_t: float = 0.5
    p_trig: float = 0.2
    delta: float = 0.3
    s_tau: float = 1.0
    delta_nt: float = 0.0
    lift: float = 0.05
    sigma_den: float = 0.5
    den_effect: float = 0.0
    independent_num: bool = False
    regime: str = ""
    a2_holds: bool = True


@dataclass(frozen=True)
class Slot:
    id: str
    design: str
    population: Literal["assigned", "triggered"]
    scale: Literal["absolute", "relative"]
    truth: str
    consequence: str


DESIGNS = (
    Design("T1-partial", "sum", GATED_REPS, 50_000, 10_000),
    Design("T1-whole", "sum", GATED_REPS, 10_000, 10_000, delta=1.0),
    Design("T4-benign", "ratio", GATED_REPS, 20_000, 10_000),
    Design("T2-baseline", "sum", 4000, 50_000, 10_000, delta_nt=0.1, a2_holds=False),
    Design("T2-powered", "sum", 12_500, 125_000, 25_000, delta_nt=0.1, a2_holds=False),
    Design("T3-binary", "variance", 4000, 20_000, 20_000, p_trig=0.1, regime="binary"),
    Design("T3-lognormal", "variance", 4000, 20_000, 20_000, p_trig=0.1, regime="lognormal"),
    Design("T3-broad", "variance", 4000, 20_000, 20_000, p_trig=0.1, regime="broad"),
    Design("T4-loaded-small", "ratio", 2000, 5000, 200, p_trig=1.0, sigma_den=1.5),
    Design("T4-loaded-large", "ratio", 2000, 5000, 2000, p_trig=1.0, sigma_den=1.5),
    Design("T4-den-moves", "ratio", 2000, 5000, 2000, p_trig=1.0, sigma_den=1.5, den_effect=0.1),
    Design("T4-triggered-heavy", "ratio", 2000, 20_000, 10_000, sigma_den=1.5),
    Design(
        "T4-independent-small",
        "ratio",
        2000,
        5000,
        200,
        p_trig=1.0,
        lift=0.0,
        sigma_den=1.5,
        independent_num=True,
    ),
    Design(
        "T4-independent-large",
        "ratio",
        2000,
        5000,
        2000,
        p_trig=1.0,
        lift=0.0,
        sigma_den=1.5,
        independent_num=True,
    ),
    Design(
        "T4-independent-triggered",
        "ratio",
        2000,
        20_000,
        10_000,
        lift=0.0,
        sigma_den=1.5,
        independent_num=True,
    ),
    Design("T5", "composition", 2000, 20_000, 20_000, p_trig=0.5, a2_holds=False),
    Design("T5b", "composition", 2000, 20_000, 20_000, p_trig=0.5),
)

_ASSIGNED_FAILURE = "Stop and escalate assigned regression; do not change assigned capability."
_TRIGGERED_FAILURE = "Refuse triggered sitewide in this design region; do not tune after sampling."
_READINGS: tuple[tuple[Literal["assigned", "triggered"], Literal["absolute", "relative"]], ...] = (
    ("assigned", "absolute"),
    ("triggered", "absolute"),
    ("assigned", "relative"),
    ("triggered", "relative"),
)
SLOTS = tuple(
    Slot(
        id=f"T1-{offset + i}",
        design=design,
        population=population,
        scale=scale,
        truth="I*" if scale == "absolute" else "rel*",
        consequence=_ASSIGNED_FAILURE if population == "assigned" else _TRIGGERED_FAILURE,
    )
    for design, offset in (("T1-partial", 1), ("T1-whole", 5))
    for i, (population, scale) in enumerate(_READINGS)
) + tuple(
    Slot(
        id=f"T4-{i + 1}",
        design="T4-benign",
        population=population,
        scale=scale,
        truth="ratio_impact*" if scale == "absolute" else "relative_ratio_impact*",
        consequence=_ASSIGNED_FAILURE if population == "assigned" else _TRIGGERED_FAILURE,
    )
    for i, (population, scale) in enumerate(_READINGS)
)

# Descriptive cells precede gates so a stopping regression retains diagnostics.
# Whole-site sums run last; this does not change any replication or decision.
EXECUTION_ORDER = tuple(d.id for d in DESIGNS if not d.id.startswith(("T1-", "T4-benign"))) + (
    "T4-benign",
    "T1-partial",
    "T1-whole",
)
DESCRIPTIVE_TRUTHS = {
    "T2": "Triggered bias = -delta_nt * N_never_triggered; assigned targets I*. Complement tau=.1 is constant; its normal-approximation power uses Var(lognormal(0,1)) = exp(2)-exp(1).",
    "T3": "Var(I_triggered)/Var(I_assigned) = p*sigma_T^2 / (p*sigma_T^2 + (1-p)*sigma_NT^2 + p*(1-p)*(mu_T-mu_NT)^2); compare within 5%, descriptively.",
    "T4": "Loaded numerator follows C.5; independent numerator is the #56 Normal(2,1) null, independent of lognormal denominator, at the matched enrollment designs. No coverage gates for these comparisons.",
    "T5": "Within H: (T0,T1)=(1,1):.45,(1,0):.05,(0,0):.50; within L: (1,1):.50,(0,1):.05,(0,0):.45. H/L equiprobable; Y0=10/0; tau=1. Equal trigger rates .50, cohort lift .50, cohort bias -.50; A1 and A2 fail.",
    "T5b": "Same membership law as T5, tau=T1. Never-triggered-under-treatment tau=0; true enrolled impact/N=.50, triggered reading/N=.25, bias/N=-.25; A1 fails while A2 holds.",
}


def arm_stats(group_id: str, y: np.ndarray, den: np.ndarray | None = None) -> ArmStats:
    """Source-compatible centered additive moments, including ratio covariance."""
    ref_y = float(y.mean())
    cy = y - ref_y
    fields: dict[str, Any] = {
        "study_id": "sitewide-calibration",
        "metric": "sum" if den is None else "ratio",
        "group_id": group_id,
        "n": len(y),
        "ref_y": ref_y,
        "cy1": float(cy.sum()),
        "cy2": float((cy * cy).sum()),
    }
    if den is not None:
        ref_den = float(den.mean())
        cd = den - ref_den
        fields.update(
            ref_den=ref_den,
            cden1=float(cd.sum()),
            cden2=float((cd * cd).sum()),
            cden3=float((cd * cd * cd).sum()),
            cyden=float((cy * cd).sum()),
        )
    return ArmStats(**fields)


def arm_moments(y: np.ndarray, den: np.ndarray | None = None) -> np.ndarray:
    """Normalize centered moments to (n, mean_y, var_mean_y, mean_d, var_mean_d, cov)."""
    n = len(y)
    if n < 2:
        return np.array([n, math.nan, math.nan, math.nan, math.nan, math.nan])
    ref_y = float(y.mean())
    cy = y - ref_y
    cy1 = float(cy.sum())
    mean_y = ref_y + cy1 / n
    var_y = (float((cy * cy).sum()) - cy1 * cy1 / n) / (n - 1) / n
    if den is None:
        return np.array([n, mean_y, var_y, 0.0, 0.0, 0.0])
    ref_d = float(den.mean())
    cd = den - ref_d
    cd1 = float(cd.sum())
    var_d = (float((cd * cd).sum()) - cd1 * cd1 / n) / (n - 1) / n
    cov = (float((cy * cd).sum()) - cy1 * cd1 / n) / (n - 1) / n
    return np.array([n, mean_y, var_y, ref_d + cd1 / n, var_d, cov])


def batched_estimates(
    control: np.ndarray,
    target: np.ndarray,
    site_num: np.ndarray,
    site_den: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Two-arm iid production algebra; invalid rows are explicitly ineligible."""
    n = control[:, 0] + target[:, 0]
    nt = target[:, 0]
    dn = target[:, 1] - control[:, 1]
    n0 = site_num - nt * dn
    eligible = (
        (control[:, 0] >= 2)
        & (target[:, 0] >= 2)
        & np.isfinite(control).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(site_num)
        & (n0 > 0)
    )
    if site_den is not None:
        dd = target[:, 3] - control[:, 3]
        d0 = site_den - nt * dd
        d1 = d0 + n * dd
        eligible &= (
            np.isfinite(site_den) & (d0 > 0) & (d1 > 0) & (control[:, 3] > 0) & (target[:, 3] > 0)
        )
    c, t = control[eligible], target[eligible]
    n, nt, dn, n0 = n[eligible], nt[eligible], dn[eligible], n0[eligible]
    if site_den is None:
        se_delta = np.sqrt(c[:, 2] + t[:, 2])
        absolute = n * dn
        se_abs = n * se_delta
        relative = absolute / n0
        se_rel = np.abs(n * site_num[eligible] / n0**2) * se_delta
    else:
        d0, d1 = d0[eligible], d1[eligible]
        n1 = n0 + n * dn
        switched = n - nt
        absolute = n1 / d1 - n0 / d0
        relative = absolute / (n0 / d0)
        gn = switched / d1 + nt / d0
        gd = -n1 * switched / d1**2 - n0 * nt / d0**2
        hn = (d0 / d1) * (switched * n0 + nt * n1) / n0**2
        hd = -(n1 / n0) * (nt * d1 + switched * d0) / d1**2
        var_abs = np.zeros(len(n))
        var_rel = np.zeros(len(n))
        for arm in (c, t):
            var_abs += gn**2 * arm[:, 2] + gd**2 * arm[:, 4] + 2 * gn * gd * arm[:, 5]
            var_rel += hn**2 * arm[:, 2] + hd**2 * arm[:, 4] + 2 * hn * hd * arm[:, 5]
        se_abs = np.sqrt(np.maximum(var_abs, 0))
        se_rel = np.sqrt(np.maximum(var_rel, 0))
    result = {"eligible": eligible}
    critical = float(norm.isf(ALPHA / 2))
    for scale, point, se in (("absolute", absolute, se_abs), ("relative", relative, se_rel)):
        for suffix, values in (
            ("", point),
            ("_se", se),
            ("_lb", point - critical * se),
            ("_ub", point + critical * se),
        ):
            full = np.full(len(eligible), np.nan)
            full[eligible] = values
            result[f"{scale}_impact{suffix}"] = full
    return result


def _selected_moments(y, den, t, trig, n_enrolled):
    y = y[:n_enrolled]
    den = None if den is None else den[:n_enrolled]
    t, trig = t[:n_enrolled], trig[:n_enrolled]
    control, target = [], []
    for membership in (np.ones(n_enrolled, dtype=bool), trig):
        c, a = ~t & membership, t & membership
        control.append(arm_moments(y[c], None if den is None else den[c]))
        target.append(arm_moments(y[a], None if den is None else den[a]))
    return np.array(control), np.array(target)


def _ratio_trial(design: Design, r: np.random.Generator):
    ns, n = design.n_site, design.n_enrolled
    den0 = r.lognormal(0, design.sigma_den, ns)
    if design.independent_num:
        num0 = r.normal(2, 1, ns)
    else:
        rate0 = np.clip(r.normal(0.2, 0.05, ns), 0.01, 0.99)
    t = np.zeros(ns, dtype=bool)
    t[:n] = r.random(n) < design.p_t
    trig = np.ones(ns, dtype=bool) if design.p_trig == 1 else r.random(ns) < design.p_trig
    eff = t & trig
    den1 = den0 * (1 + design.den_effect)
    den = np.where(eff, den1, den0)
    if design.independent_num:
        num1 = num0 + design.lift
    else:
        noise = r.normal(0, 0.1, ns)
        num0 = den0 * rate0 + noise * den0
        num1 = den1 * (rate0 + design.lift) + noise * den1
    y = np.where(eff, num1, num0)
    volume = float(y.sum())
    volume_den = float(den.sum())
    ship = trig.copy()
    ship[n:] = False
    v0n = volume - float((num1 - num0)[eff].sum())
    v0d = volume_den - float((den1 - den0)[eff].sum())
    v1n = v0n + float((num1 - num0)[ship].sum())
    v1d = v0d + float((den1 - den0)[ship].sum())
    truth_abs = v1n / v1d - v0n / v0d
    truth_rel = truth_abs / (v0n / v0d)
    return y, den, t, trig, volume, volume_den, truth_abs, truth_rel


def _trial(design: Design, rep: int):
    """Draw calendar outcomes, fixed membership, assignment and realized truths."""
    r = np.random.default_rng(SEED_START + rep)
    ns, n = design.n_site, design.n_enrolled
    diagnostics: dict[str, float] = {}
    den = None
    if design.kind == "sum":
        y0 = r.lognormal(0, 1, ns)
        tau = design.delta + design.s_tau * r.normal(size=ns)
        t = np.zeros(ns, dtype=bool)
        t[:n] = r.random(n) < design.p_t
        trig = r.random(ns) < design.p_trig
        tau = np.where(trig, tau, design.delta_nt)
        y = y0 + tau * t
        volume = float(y.sum())
        truth_abs = float(tau[:n].sum())
        truth_rel = truth_abs / float(volume - tau[t].sum())
        if design.delta_nt:
            c, a = ~t[:n] & ~trig[:n], t[:n] & ~trig[:n]
            cm, tm = arm_moments(y[:n][c]), arm_moments(y[:n][a])
            comp_delta = tm[1] - cm[1]
            comp_se = math.sqrt(tm[2] + cm[2])
            diagnostics = {
                "complement_effect": comp_delta,
                "complement_reject": float(abs(comp_delta) > norm.isf(ALPHA / 2) * comp_se),
                "complement_n_control": cm[0],
                "complement_n_treatment": tm[0],
                "expected_triggered_bias": -design.delta_nt * float((~trig[:n]).sum()),
            }
    elif design.kind == "ratio":
        y, den, t, trig, volume, volume_den, truth_abs, truth_rel = _ratio_trial(design, r)
    elif design.kind == "variance":
        t = r.random(n) < design.p_t
        trig = r.random(n) < design.p_trig
        if design.regime == "binary":
            y = np.where(trig, (r.random(n) < 0.1).astype(float), np.zeros(n))
        elif design.regime == "lognormal":
            y = np.where(trig, r.lognormal(0, 1, n), np.zeros(n))
        else:
            y = np.where(trig, r.normal(0, 1, n), r.normal(0, 1, n))
        volume = float(y.sum()) + VARIANCE_BACKGROUND
        truth_abs = truth_rel = 0.0
    else:
        high = r.random(n) < 0.5
        u = r.random(n)
        t0 = u < 0.50
        t1 = np.where(high, u < 0.45, u < 0.55)
        y0 = 10.0 * high
        tau = t1.astype(float) if design.a2_holds else np.ones(n)
        t = r.random(n) < design.p_t
        trig = np.where(t, t1, t0)
        y = y0 + tau * t
        volume = float(y.sum())
        truth_abs = float(tau.sum())
        truth_rel = truth_abs / float(y0.sum())
        rate_c, rate_t = float(trig[~t].mean()), float(trig[t].mean())
        pooled = float(trig.mean())
        rate_se = math.sqrt(pooled * (1 - pooled) * (1 / t.sum() + 1 / (~t).sum()))
        diagnostics = {
            "trigger_rate_control": rate_c,
            "trigger_rate_treatment": rate_t,
            "srm_reject": float(abs(rate_t - rate_c) > norm.isf(ALPHA / 2) * rate_se),
            "true_impact_per_enrolled_unit": truth_abs / n,
            "never_triggered_mean_effect": float(tau[~t1].mean()),
        }
    control, target = _selected_moments(y, den, t, trig, n)
    return (
        control,
        target,
        volume,
        None if den is None else volume_den,
        truth_abs,
        truth_rel,
        diagnostics,
    )


def _run_chunk(args):
    design, start, stop = args
    trials = [_trial(design, rep) for rep in range(start, stop)]
    controls = np.array([x[0] for x in trials])
    targets = np.array([x[1] for x in trials])
    volumes = np.array([x[2] for x in trials])
    dens = np.array([x[3] for x in trials]) if design.kind == "ratio" else None
    records: dict[str, np.ndarray] = {
        "truth.absolute": np.array([x[4] for x in trials]),
        "truth.relative": np.array([x[5] for x in trials]),
    }
    for key in trials[0][6]:
        records[key] = np.array([x[6][key] for x in trials])
    for i, population in enumerate(("assigned", "triggered")):
        estimates = batched_estimates(controls[:, i], targets[:, i], volumes, dens)
        records[f"{population}.eligible"] = estimates["eligible"]
        records[f"{population}.cohort_effect"] = targets[:, i, 1] - controls[:, i, 1]
        records[f"{population}.scaled_units"] = targets[:, i, 0] + controls[:, i, 0]
        for scale in ("absolute", "relative"):
            prefix = f"{population}.{scale}"
            records[f"{prefix}.estimate"] = estimates[f"{scale}_impact"]
            records[f"{prefix}.error"] = estimates[f"{scale}_impact"] - records[f"truth.{scale}"]
            records[f"{prefix}.hit"] = (
                estimates[f"{scale}_impact_lb"] <= records[f"truth.{scale}"]
            ) & (records[f"truth.{scale}"] <= estimates[f"{scale}_impact_ub"])
    return records


def _mean_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    return {
        "mean": float(finite.mean()) if len(finite) else None,
        "mcse": float(finite.std(ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else None,
        "replications": len(finite),
    }


def _summarize(design: Design, records: dict[str, np.ndarray], reps: int):
    result: dict[str, Any] = {"id": design.id, "attempted": reps, "coverage": {}, "summaries": {}}
    for population, scale in _READINGS:
        key = f"{population}.{scale}"
        eligible = records[f"{population}.eligible"]
        k = int(records[f"{key}.hit"][eligible].sum())
        count = int(eligible.sum())
        result["coverage"][key] = {
            "hits": k,
            "eligible": count,
            "refused": reps - count,
            "coverage": k / count if count else None,
            "lower_bound": coverage_lower_bound(k, count, ETA) if count else None,
            "upper_bound": binomial_error_upper_bound(k, count, ETA) if count else None,
            "miss_upper_bound": binomial_error_upper_bound(count - k, count, ETA)
            if count
            else None,
            "mcse": mcse(k / count, count) if count else None,
        }
    for key, values in records.items():
        if not key.endswith((".hit", ".eligible")):
            result["summaries"][key] = _mean_summary(values)
        elif key.endswith(".eligible"):
            result["summaries"][key] = _mean_summary(values.astype(float))
    differences = records["triggered.absolute.estimate"] - records["assigned.absolute.estimate"]
    result["summaries"]["reading_difference.absolute"] = _mean_summary(differences)
    if design.id.startswith("T2-"):
        complement_per_arm = design.n_enrolled * (1 - design.p_trig) / 2
        se = math.sqrt(2 * (math.exp(2) - math.exp(1)) / complement_per_arm)
        z = float(norm.isf(ALPHA / 2))
        result["normal_approximation_complement_power"] = float(
            norm.sf(z - design.delta_nt / se) + norm.cdf(-z - design.delta_nt / se)
        )
        result["complement_rejection"] = _proportion_summary(records["complement_reject"])
    if design.kind == "variance":
        p = design.p_trig
        if design.regime == "binary":
            st, sn, mt, mn = 0.09, 0.0, 0.1, 0.0
        elif design.regime == "lognormal":
            st, sn, mt, mn = math.exp(2) - math.exp(1), 0.0, math.exp(0.5), 0.0
        else:
            st, sn, mt, mn = 1.0, 1.0, 0.0, 0.0
        formula = p * st / (p * st + (1 - p) * sn + p * (1 - p) * (mt - mn) ** 2)
        a = records["assigned.absolute.estimate"]
        b = records["triggered.absolute.estimate"]
        ratio = float(b.var(ddof=1) / a.var(ddof=1))
        # Paired influence-function MCSE accounts for the shared draws/readings.
        va, vb = a.var(ddof=1), b.var(ddof=1)
        influence = ((b - b.mean()) ** 2 - ratio * (a - a.mean()) ** 2) / va
        result["variance_ratio"] = {
            "formula": formula,
            "observed": ratio,
            "mcse": float(influence.std(ddof=1) / math.sqrt(reps)),
            "relative_error": ratio / formula - 1,
            "within_five_percent": abs(ratio / formula - 1) <= 0.05,
            "assigned_variance": float(va),
            "triggered_variance": float(vb),
        }
    if design.kind == "composition":
        result["srm_rejection"] = _proportion_summary(records["srm_reject"])
        result["summaries"]["triggered_reading_per_enrolled_unit"] = _mean_summary(
            records["triggered.absolute.estimate"] / design.n_enrolled
        )
        result["summaries"]["triggered_bias_per_enrolled_unit"] = _mean_summary(
            records["triggered.absolute.error"] / design.n_enrolled
        )
    return result


def _proportion_summary(values):
    hits, reps = int(values.sum()), len(values)
    return {
        "rate": hits / reps,
        "mcse": mcse(hits / reps, reps),
        "lower_bound": coverage_lower_bound(hits, reps, 0.025),
        "upper_bound": binomial_error_upper_bound(hits, reps, 0.025),
    }


def _run_design(design: Design, reps: int, executor: ProcessPoolExecutor | None):
    chunks = [(design, start, min(start + 256, reps)) for start in range(0, reps, 256)]
    completed = (
        list(map(_run_chunk, chunks))
        if executor is None
        else list(executor.map(_run_chunk, chunks))
    )
    records = {key: np.concatenate([chunk[key] for chunk in completed]) for key in completed[0]}
    return _summarize(design, records, reps)


def _check_workers(workers: int):
    if not 1 <= workers <= 6:
        raise ValueError("workers must be between 1 and 6")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"


def run_design(design: Design, *, reps: int, workers: int = 1) -> dict[str, Any]:
    """Run fixed seeds once each; disclose rather than replace refused replications."""
    if reps < 2:
        raise ValueError("at least two replications are required")
    _check_workers(workers)
    if workers == 1:
        return _run_design(design, reps, None)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        return _run_design(design, reps, executor)


def manifest() -> dict[str, Any]:
    return {
        "gated_reps": GATED_REPS,
        "family_size": 67,
        "family_alpha": 0.01,
        "eta": ETA,
        "coverage_floor": COVERAGE_FLOOR,
        "alpha": ALPHA,
        "seed_start": SEED_START,
        "seed_rule": "1000 + replication_index within every design",
        "eligibility": "Each frozen seed is attempted once; no replacement or tuning.",
        "variance_background": VARIANCE_BACKGROUND,
        "execution_order": list(EXECUTION_ORDER),
        "designs": [asdict(d) for d in DESIGNS],
        "slots": [asdict(s) for s in SLOTS],
        "descriptive_truths": DESCRIPTIVE_TRUTHS,
        "sentinel": {"reps": SENTINEL_REPS, "eta": SENTINEL_ETA, "floor": SENTINEL_FLOOR},
    }


def _gates(cells: list[dict[str, Any]], campaign: str):
    lookup = {c["id"]: c for c in cells}
    gates = []
    for slot in SLOTS:
        if slot.design not in lookup:
            gates.append({**asdict(slot), "decision": "NOT_RUN"})
            continue
        cell = lookup[slot.design]
        coverage = cell["coverage"][f"{slot.population}.{slot.scale}"]
        if campaign == "smoke":
            decision = "SMOKE_ONLY"
        elif coverage["eligible"] != GATED_REPS:
            decision = "INELIGIBLE"
        else:
            decision = "PASS" if coverage["lower_bound"] >= COVERAGE_FLOOR else "FAIL"
        gates.append({**asdict(slot), **coverage, "decision": decision})
    return gates


def _write_report(output: Path, report: dict[str, Any]):
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Triggered sitewide calibration",
        "",
        f"Status: {report['status']}. Wall time: {report['wall_seconds']:.3f} seconds.",
        f"Workers: {report['workers']}; replication seed = 1000 + rep.",
        f"Full gate: R={GATED_REPS}, eta={ETA}, lower bound >= {COVERAGE_FLOOR}.",
        "",
        "| Slot | Population | Scale | Eligible | Coverage | CP lower | CP upper | Decision |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for gate in report["gates"]:
        if gate["decision"] == "NOT_RUN":
            lines.append(
                f"| {gate['id']} | {gate['population']} | {gate['scale']} | - | - | - | - | NOT_RUN |"
            )
        else:
            numbers = [
                "-" if gate[key] is None else f"{gate[key]:.6f}"
                for key in ("coverage", "lower_bound", "upper_bound")
            ]
            lines.append(
                f"| {gate['id']} | {gate['population']} | {gate['scale']} | {gate['eligible']} "
                f"| {' | '.join(numbers)} | {gate['decision']} |"
            )
        if gate["decision"] in ("FAIL", "INELIGIBLE"):
            lines.append(f"\n{gate['id']}: {gate['consequence']}\n")
    lines += [
        "",
        "## Descriptive results",
        "",
        "All diagnostics below are descriptive, not additional family gates.",
    ]
    for cell in report["cells"]:
        lines += [
            "",
            f"### {cell['id']}",
            "",
            "```json",
            json.dumps(cell, indent=2, allow_nan=False),
            "```",
        ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run_campaign(
    campaign: Literal["smoke", "full"],
    output: Path,
    *,
    workers: int = 6,
) -> dict[str, Any]:
    """Write the frozen manifest first; stop immediately after an assigned failed design."""
    _check_workers(workers)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest(), indent=2, allow_nan=False) + "\n")
    started = time.monotonic()
    report: dict[str, Any] = {
        "campaign": campaign,
        "workers": workers,
        "status": "RUNNING",
        "cells": [],
        "gates": [],
        "wall_seconds": 0.0,
    }
    executor = (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        if workers > 1
        else None
    )
    try:
        for design_id in EXECUTION_ORDER:
            design = next(d for d in DESIGNS if d.id == design_id)
            reps = design.reps if campaign == "full" else SMOKE_REPS
            cell = _run_design(design, reps, executor)
            report["cells"].append(cell)
            report["gates"] = _gates(report["cells"], campaign)
            report["wall_seconds"] = time.monotonic() - started
            failed = [g for g in report["gates"] if g["decision"] in ("FAIL", "INELIGIBLE")]
            assigned_failure = any(g["population"] == "assigned" for g in failed)
            report["status"] = "ASSIGNED_REGRESSION" if assigned_failure else "RUNNING"
            _write_report(output, report)
            print(f"{design.id}: R={reps}, elapsed={report['wall_seconds']:.1f}s", flush=True)
            if assigned_failure:
                break
        if report["status"] != "ASSIGNED_REGRESSION":
            failed = any(g["decision"] in ("FAIL", "INELIGIBLE") for g in report["gates"])
            report["status"] = "TRIGGERED_REGION_REFUSAL" if failed else "COMPLETE"
    finally:
        if executor is not None:
            executor.shutdown()
    report["wall_seconds"] = time.monotonic() - started
    _write_report(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    args = parser.parse_args(argv)
    report = run_campaign(args.campaign, args.output, workers=args.workers)
    print(
        f"{report['status']}: {report['wall_seconds']:.3f}s; report={args.output / 'report.json'}"
    )
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
