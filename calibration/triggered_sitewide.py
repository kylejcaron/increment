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
baseline, and positive ratio component means, D0, D1 and N0. The delta-method
interval conditions on the realized site volume and membership; it omits
future-traffic and estimated-reach uncertainty. Coverage targets are the
realized finite-population I* = sum(tau_enrolled), V0* = V - sum(tau_treated)
and rel* = I*/V0*, not a superpopulation mean. Ratio truths apply the same
potential-outcome accounting to both components. Never-triggered and
non-enrolled outcomes remain in V even for the triggered reading.

T3 retains the C.2 outcomes and adds a fixed unaffected site background of
1,000,000 solely to admit a positive baseline for the broad zero-mean metric.
Only its absolute-impact variances are compared; these do not depend on V.

The manifest below is frozen before sampling. Every design uses independent
replication seeds 1000 + rep; readings within a design are paired. Full gates
share the 67-statistic family allocation with other certification campaigns.
No rejected replication is replaced: failing to attain the frozen eligible
count refuses the region. An assigned failure stops the campaign and requires
escalation; a triggered failure requires refusing that triggered design region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
