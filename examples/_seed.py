"""Synthetic warehouse seed shared by the runnable examples.

``examples/definitions/fact_sources.yaml`` declares exactly one fact source: the
relation ``analytics.event_log``.  Every example that analyses anything needs
that relation to exist first, so this module generates a deterministic synthetic
version of it.

The point of generating rather than shipping a CSV is that the *shape* of the
table is the hard part of adopting the semantic layer.  This is the mapping the
definitions expect:

==============================  ==========================================
column                          role in ``examples/definitions/``
==============================  ==========================================
``event_at``                    ``FactSource.timestamp_column``
``user_id``                     entity, and ``Experiment.unit``
``session_id``                  second declared entity
``event``                       fact discriminator: ``page_view``,
                                ``purchase``, ``session_start``,
                                ``session_end``
``revenue``                     value column of the ``purchase`` fact
``duration_s``                  value column of the ``session_end`` fact
``country_code``                property ``country``
``device_type``                 property ``platform``
``plan``                        property ``plan_tier``
``experiment_id``, ``group_id`` read by the fact-based ``first_page_view``
                                exposure to assign units to arms
==============================  ==========================================

Only the enrolling ``page_view`` row carries ``experiment_id``/``group_id``.
Stamping the same assignment onto every later row of the unit would be harmless
-- ``first_exposures`` would still see a single distinct group and still resolve
the exposure to ``min(event_at)`` -- but it would be pure redundancy, and it
would blur the exposure semantics: exactly one row per unit is the first page
view in the window, and that row is the one that carries the assignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

if TYPE_CHECKING:
    from ibis.backends.duckdb import Backend

# ---------------------------------------------------------------------------
# Experiment geometry -- every constant mirrors examples/definitions/
# ---------------------------------------------------------------------------

#: The experiment the seeded population belongs to (experiments.yaml).
EXPERIMENT_ID = "new_onboarding_v2"

#: ``new_onboarding_v2`` runs 2025-01-15 .. 2025-02-15.
_EXPERIMENT_START = np.datetime64("2025-01-15T00:00:00", "us")

#: Enrollment ends on Jan 31, keeping every unit inside the 14-day observation
#: horizon. Each metric also caps censoring at its ``data_as_of``; generated
#: facts can end earlier, especially session data, so that gap is intentional.
_ENROLLMENT_DAYS = 17

#: Post-exposure horizon.  Covers ``purchase_rate``'s 14-day conversion window
#: and reaches past ``d7_retention``'s 7-day threshold.
_POST_DAYS = 14

#: ``avg_session_duration.window_days``.
_SESSION_WINDOW_DAYS = 7

#: ``d7_retention.threshold_days`` -- a unit only counts as retained if it is
#: active on or after this day, so retention page views start here.
_RETENTION_THRESHOLD_DAYS = 7

#: Late-return injection lands just past the bounded ``[7, 14)`` retention
#: band, but before the experiment ends, so bounded and unbounded metrics
#: can demonstrate their different treatment of the same event.
_LATE_RETURN_TENURE_LOW = 14
_LATE_RETURN_TENURE_HIGH = 16  # exclusive upper bound for rng.integers

#: Share of units that otherwise never return in-band which get a late
#: return injected when ``with_late_returns=True``.
_LATE_RETURN_FRACTION = 0.40

#: ``Experiment.n_pre_periods`` -- the CUPED lookback, in days.
_PRE_DAYS = 14

_HOUR = 3600
_DAY = 86_400

_SCHEMA = pa.schema(
    [
        ("event_at", pa.timestamp("us")),
        ("user_id", pa.string()),
        ("session_id", pa.string()),
        ("event", pa.string()),
        ("revenue", pa.float64()),
        ("duration_s", pa.float64()),
        ("country_code", pa.string()),
        ("device_type", pa.string()),
        ("plan", pa.string()),
        ("experiment_id", pa.string()),
        ("group_id", pa.string()),
    ]
)


def _inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _running_index(counts: np.ndarray) -> np.ndarray:
    """0, 1, 2, ... within each unit's run of ``np.repeat(units, counts)``.

    ``np.repeat`` lays a unit's rows out contiguously, so a global ``arange``
    minus each run's start offset counts the rows within every unit.  This is
    what gives sessions a collision-free per-unit ``session_id``.
    """
    return np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def seed_event_log(
    con: Backend,
    *,
    n_units: int = 4000,
    seed: int = 2025,
    with_pre_period: bool = False,
    with_late_returns: bool = False,
) -> None:
    """Create ``analytics.event_log`` in *con* and fill it with one experiment.

    The population is a single ``new_onboarding_v2`` cohort with a baked-in
    treatment effect, so an unadjusted analysis reports a clearly positive lift
    on every metric the experiment declares.  Those effects are deliberately far
    larger than anything a real onboarding test would produce -- the point is
    unambiguous demo output, not realism.  A real test at this sample size would
    usually come back inconclusive.

    Parameters
    ----------
    con :
        An ibis DuckDB backend.  The ``analytics`` schema is created if needed
        (DuckDB will not auto-create it) and any existing ``event_log`` is
        replaced.
    n_units :
        Number of randomised users.  4000 keeps every metric's interval clear of
        zero across seeds (2000 is knife-edge: roughly one seed in ten leaves
        ``avg_session_duration`` inconclusive), and it costs nothing -- seeding
        plus the full analysis still runs in well under a second.
    seed :
        RNG seed.  Fixed by default so the printed numbers are reproducible.
    with_pre_period :
        Also emit activity in the 14 days *before* each unit's exposure.  Off by
        default -- the end-to-end example does not need it.  ``examples/cuped.py``
        turns it on: CUPED builds its covariate from pre-exposure events and
        only reduces variance when that covariate correlates with the outcome.
        That correlation is manufactured here, by driving pre- and post-exposure
        behaviour from the same per-unit latent propensity.

        The pre-period draws come from their own RNG stream, so adding
        pre-period events does not reshuffle unrelated draws.  The CUPED example
        also opts into a stronger shared propensity signal, intentionally making
        its post-period outcome distribution different from examples without a
        pre-period.  Both methods in that notebook still use the exact same
        generated event log.

        How much CUPED can buy here is capped by the metric more than by the
        density of the pre-period.  For a binary outcome the covariate's
        correlation with the outcome cannot exceed ``sd(p) / sd(y)`` -- about
        0.33 for ``purchase_rate`` -- so even a perfect covariate would narrow
        its interval by only ~6 %.  The continuous session metric has more
        predictable variance, and the tuned CUPED example narrows its interval
        by about 41 %.  A CUPED demo should therefore lead with
        ``avg_session_duration``.
    with_late_returns :
        Also emit one extra "page view" at tenure 14-15 for a share of the
        units that otherwise never return in-band.  Off by default -- the
        end-to-end example does not need it.  Draws come from their own RNG
        stream, so enabling this leaves every other event bit-identical: the
        same isolation guarantee ``with_pre_period`` documents above.

        ``d7_retention`` is bounded (``threshold_days: [7, 14]``), and the
        shipped seed never emits a page view past tenure 13 -- so an
        unbounded metric sharing its 7-day threshold (``threshold_days: 7``)
        computes IDENTICAL numbers to the bounded one; there is no seeded
        event past the band close for the two shapes to disagree on.
        Turning this on injects a page view, just past the band's close, for
        `_LATE_RETURN_FRACTION` (40 %) of the units currently scoring 0 under
        both metrics: those units keep scoring 0 under the bounded metric
        (the return lands after its band has already shut) but flip to 1
        under the unbounded one (its band never closes), so the two metrics'
        whole-window rates now differ measurably -- enough to build an
        unbounded-vs-bounded retention example against.
    """
    # Independent child streams keep the seed reproducible while preventing a
    # new draw in one generation block from reshuffling every other block.
    # The population stream also owns assignment, properties, and exposures.
    rng_population, rng_purchase, rng_session, rng_retention, rng_pre, rng_late = [
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(6)
    ]
    units = np.arange(n_units)
    user_id = np.array([f"u{i:05d}" for i in range(n_units)])

    # One latent propensity per unit: heavy users buy more, browse more and stay
    # longer.  This is what makes a pre-period covariate informative.
    propensity = rng_population.normal(size=n_units)

    # Balanced 50/50 assignment rather than coin flips, so the arms have equal n
    # and the printed lift isn't muddied by an accidental split imbalance.
    is_treated = np.zeros(n_units, dtype=bool)
    is_treated[rng_population.permutation(n_units)[: n_units // 2]] = True
    group_id = np.where(is_treated, "treatment", "control")
    treated = is_treated.astype(float)
    # CUPED's example uses a stronger shared signal so the variance reduction is
    # visible; examples without a pre-period keep the original data-generating
    # relationship.
    propensity_signal = 1.8 if with_pre_period else 1.0

    country_code = rng_population.choice(
        ["US", "GB", "DE", "CA"], size=n_units, p=[0.60, 0.20, 0.12, 0.08]
    )
    # ``avg_session_duration`` filters ``platform == "web"``, so the population
    # is overwhelmingly web.  The few mobile-only units contribute 0 seconds to
    # that metric -- realistic, and it keeps the filter honest instead of inert.
    device_type = rng_population.choice(
        ["web", "ios", "android"], size=n_units, p=[0.94, 0.03, 0.03]
    )
    plan = rng_population.choice(["free", "pro", "enterprise"], size=n_units, p=[0.70, 0.25, 0.05])

    # Seconds since _EXPERIMENT_START of each unit's exposure: 09:00 on its
    # enrollment day.  Every other event is an offset from this, which is what
    # guarantees post-exposure events land after the exposure itself (the panel
    # join drops events with ``ts < first_exposure_ts``).
    exposure_second = rng_population.integers(0, _ENROLLMENT_DAYS, size=n_units) * _DAY + 9 * _HOUR

    def chunk(
        unit: np.ndarray,
        second: np.ndarray,
        session_key: np.ndarray,
        event: str,
        *,
        revenue: np.ndarray | None = None,
        duration_s: np.ndarray | None = None,
        is_exposure: bool = False,
    ) -> pa.Table:
        """One event type as an Arrow table shaped like ``analytics.event_log``.

        *unit* selects the user behind each row, so every per-unit property is
        broadcast out of the population arrays above.
        """
        n = unit.size
        return pa.table(
            [
                pa.array(_EXPERIMENT_START + second.astype("timedelta64[s]")),
                pa.array(user_id[unit], type=pa.string()),
                pa.array(
                    [f"{u}-s{k}" for u, k in zip(user_id[unit], session_key, strict=True)],
                    type=pa.string(),
                ),
                pa.array(np.full(n, event), type=pa.string()),
                pa.nulls(n, pa.float64()) if revenue is None else pa.array(revenue, pa.float64()),
                pa.nulls(n, pa.float64())
                if duration_s is None
                else pa.array(duration_s, pa.float64()),
                pa.array(country_code[unit], type=pa.string()),
                pa.array(device_type[unit], type=pa.string()),
                pa.array(plan[unit], type=pa.string()),
                pa.array(np.full(n, EXPERIMENT_ID), type=pa.string())
                if is_exposure
                else pa.nulls(n, pa.string()),
                pa.array(group_id[unit], type=pa.string())
                if is_exposure
                else pa.nulls(n, pa.string()),
            ],
            schema=_SCHEMA,
        )

    chunks: list[pa.Table] = []

    # ── Exposure: the single page view that enrolls each unit ──────────────
    chunks.append(
        chunk(units, exposure_second, exposure_second // _HOUR, "page_view", is_exposure=True)
    )

    # ── purchase_rate: one purchase inside the 14-day conversion window ────
    # Baseline is ~27 % conversion; treatment adds 0.50 on the logit scale,
    # i.e. roughly a third more converters.
    p_purchase = _inv_logit(-1.1 + 0.75 * propensity + 0.50 * treated)
    buyer = np.flatnonzero(rng_purchase.random(n_units) < p_purchase)
    buy_second = (
        exposure_second[buyer]
        + rng_purchase.integers(1, _POST_DAYS, size=buyer.size) * _DAY
        + 3 * _HOUR
    )
    chunks.append(
        chunk(
            buyer,
            buy_second,
            buy_second // _HOUR,
            "purchase",
            revenue=np.round(rng_purchase.lognormal(3.2, 0.7, size=buyer.size), 2),
        )
    )

    # ── avg_session_duration: sessions inside the 7-day mean window ────────
    # Treatment lifts both the session count (+11 %) and the session length
    # (+13 %), so the per-unit daily mean rises by roughly 25 %.
    session_count = rng_session.poisson(
        np.exp(0.55 + 0.35 * propensity_signal * propensity + 0.10 * treated)
    )
    session_unit = np.repeat(units, session_count)
    session_second = (
        exposure_second[session_unit]
        + rng_session.integers(0, _SESSION_WINDOW_DAYS, size=session_unit.size) * _DAY
        # 10:00-20:00, i.e. always after the 09:00 exposure on day 0
        + rng_session.integers(1, 12, size=session_unit.size) * _HOUR
    )
    duration_s = np.round(
        rng_session.lognormal(
            4.9
            + 0.25 * propensity_signal * propensity[session_unit]
            + 0.12 * treated[session_unit],
            0.5,
        )
    )
    # Session halves share a per-unit running ID; start-hour keys collide over seven days.
    # Pre-period IDs use a ``p`` prefix. Attribution uses the end day; filtering uses the start.
    # Starts before 20:00 and durations under four hours keep both days equal.
    # Preserve this invariant when changing either generation parameter.
    session_key = _running_index(session_count)
    chunks.append(chunk(session_unit, session_second, session_key, "session_start"))
    chunks.append(
        chunk(
            session_unit,
            session_second + duration_s.astype(np.int64),
            session_key,
            "session_end",
            duration_s=duration_s,
        )
    )

    # ── d7_retention: a page view on or after day 7 post-exposure ──────────
    p_retained = _inv_logit(0.15 + 0.60 * propensity + 0.45 * treated)
    retained = np.flatnonzero(rng_retention.random(n_units) < p_retained)
    return_second = (
        exposure_second[retained]
        + rng_retention.integers(_RETENTION_THRESHOLD_DAYS, _POST_DAYS, size=retained.size) * _DAY
        + 6 * _HOUR
    )
    chunks.append(chunk(retained, return_second, return_second // _HOUR, "page_view"))

    # ── Optional late returns: an unbounded-vs-bounded discriminator ───────
    if with_late_returns:
        never_returned = np.setdiff1d(units, retained, assume_unique=True)
        late_returner = never_returned[rng_late.random(never_returned.size) < _LATE_RETURN_FRACTION]
        late_return_second = (
            exposure_second[late_returner]
            + rng_late.integers(
                _LATE_RETURN_TENURE_LOW, _LATE_RETURN_TENURE_HIGH, size=late_returner.size
            )
            * _DAY
            + 6 * _HOUR
        )
        chunks.append(
            chunk(
                late_returner,
                late_return_second,
                late_return_second // _HOUR,
                "page_view",
            )
        )

    # ── Optional pre-period activity: the CUPED covariate ──────────────────
    if with_pre_period:

        def pre_second(unit: np.ndarray) -> np.ndarray:
            """Noon, 1-14 days before each unit's exposure.

            ``unit_totals`` scopes the covariate to
            ``[first_exposure_date - n_pre_periods, first_exposure_ts)``, so
            these have to be strictly before the exposure and no further back
            than 14 days.
            """
            days_back = rng_pre.integers(1, _PRE_DAYS + 1, size=unit.size)
            return exposure_second[unit] - days_back * _DAY + 3 * _HOUR

        # Each metric uses the same fact and filters as its outcome; no
        # treatment term enters the pre-period covariate before assignment.
        pre_buyer = np.repeat(units, rng_pre.poisson(np.exp(-0.90 + 0.70 * propensity)))
        pre_buy_second = pre_second(pre_buyer)
        chunks.append(
            chunk(
                pre_buyer,
                pre_buy_second,
                pre_buy_second // _HOUR,
                "purchase",
                revenue=np.round(rng_pre.lognormal(3.2, 0.7, size=pre_buyer.size), 2),
            )
        )

        pre_session_count = rng_pre.poisson(np.exp(2.00 + 0.35 * propensity_signal * propensity))
        pre_session_unit = np.repeat(units, pre_session_count)
        pre_session_second = pre_second(pre_session_unit)
        chunks.append(
            chunk(
                pre_session_unit,
                pre_session_second,
                np.char.add("p", _running_index(pre_session_count).astype(str)),
                "session_end",
                duration_s=np.round(
                    rng_pre.lognormal(
                        4.9 + 0.25 * propensity_signal * propensity[pre_session_unit],
                        0.20,
                    )
                ),
            )
        )

        # These page views carry no experiment_id, so they can never be mistaken
        # for an exposure: the exposure filter compares experiment_id against
        # the experiment name, and NULL never matches.
        pre_viewer = np.repeat(units, rng_pre.poisson(np.exp(0.30 + 0.60 * propensity)))
        pre_view_second = pre_second(pre_viewer)
        chunks.append(chunk(pre_viewer, pre_view_second, pre_view_second // _HOUR, "page_view"))

    event_log = pa.concat_tables(chunks).sort_by("event_at")

    con.raw_sql("CREATE SCHEMA IF NOT EXISTS analytics")
    con.create_table("event_log", event_log, database="analytics", overwrite=True)
