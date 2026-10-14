"""Deterministic randomized roster and serially correlated panel witnesses."""

import numpy as np
import pandas as pd

from increment.longitudinal import LongitudinalCohort, LongitudinalDesign


def panel_data(*, unbalanced=False, clustered=False, adjusted=False, heterogeneous=False):
    rng = np.random.default_rng(80)
    cohorts = [
        LongitudinalCohort(
            name="early",
            exposure=0,
            pre_periods=(-2, -1),
            post_periods=(0, 1),
            post_weights=(0.25, 0.75),
        )
    ]
    if unbalanced:
        cohorts.append(
            LongitudinalCohort(name="late", exposure=2, pre_periods=(0, 1), post_periods=(2, 3, 4))
        )
    design = LongitudinalDesign(
        control_group="C",
        treatment_group="T",
        cohorts=tuple(cohorts),
        independence_grain="cluster" if clustered else "unit",
    )
    roster, observations = [], []
    for j, c in enumerate(cohorts):
        for i in range(48):
            arm = "C" if i < 24 else "T"
            unit = f"u{j}_{i:02}"
            cluster = f"g{j}_{i // 2:02}" if clustered else unit
            x = rng.normal()
            kind = "a" if i % 3 else "b"
            roster.append(
                {
                    "unit": unit,
                    "group": arm,
                    "cohort": c.name,
                    "exposure": c.exposure,
                    "cluster": cluster,
                    "x": x,
                    "kind": kind,
                }
            )
            intercept = rng.normal(scale=5)
            noise = rng.normal()
            for period in (*c.pre_periods, *c.post_periods):
                assert isinstance(period, int) and isinstance(c.exposure, int)
                assert isinstance(c.pre_periods[0], int)
                noise = 0.7 * noise + rng.normal()
                shift = (1 + 0.3 * x + (period - c.exposure) * 0.4) if heterogeneous else 1.0
                y = intercept + period * 1.5 + noise
                if adjusted:
                    y += (2 * x + (kind == "b") * 1.2) * (period - c.pre_periods[0])
                if arm == "T" and period in c.post_periods:
                    y += shift
                observations.append({"unit": unit, "time": period, "outcome": y})
    return pd.DataFrame(observations), pd.DataFrame(roster), design
