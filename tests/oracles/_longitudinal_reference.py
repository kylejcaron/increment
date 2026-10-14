"""Independent dense panel reference: unit FE and saturated arm/calendar slopes.

Does not import any Increment source builder, encoder or estimator. CR1 uses
G/(G-1); time and pre-treatment covariate contrasts include EVERY pre period.
"""

import numpy as np


def dense_panel_reference(outcomes, roster, design, *, adjusted=False, clustered=False):
    units = roster.unit.tolist()
    rows = outcomes.merge(roster, on="unit", validate="many_to_one")
    unit_columns = [(rows.unit == u).to_numpy(dtype=float) for u in units]
    xs = list(unit_columns)
    weights = [0.0] * len(units)
    for c in design.cohorts:
        members = roster[roster.cohort == c.name]
        share = len(members) / len(roster)
        # Independent encoding uses a different reference category and scale.
        features = [(None, 0.0)]
        if adjusted:
            features.extend([("x", members.x.mean()), ("is_b", (members.kind == "b").mean())])
        q = {t: -1 / len(c.pre_periods) for t in c.pre_periods}
        q.update(dict(zip(c.post_periods, c.post_weights, strict=True)))
        first = c.pre_periods[0]
        for arm, sign in (("C", -1), ("T", 1)):
            for period in (*c.pre_periods, *c.post_periods):
                if period == first:
                    continue
                indicator = (
                    (rows.cohort == c.name) & (rows.group == arm) & (rows.time == period)
                ).to_numpy(dtype=float)
                for feature, center in features:
                    values = (
                        np.ones(len(rows))
                        if feature is None
                        else (
                            rows.x.to_numpy()
                            if feature == "x"
                            else (rows.kind == "b").to_numpy(dtype=float)
                        )
                        - center
                    )
                    xs.append(indicator * values)
                    weights.append(share * sign * q[period] if feature is None else 0.0)
    x = np.column_stack(xs)
    y = rows.outcome.to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    # Solve contrast influence directly instead of rebuilding production Gram scores.
    direction = np.linalg.lstsq(x.T @ x, np.asarray(weights), rcond=None)[0]
    row_scores = (x @ direction) * residual
    grain = rows.cluster if clustered else rows.unit
    scores = np.array([row_scores[(grain == g).to_numpy()].sum() for g in sorted(set(grain))])
    variance = len(scores) / (len(scores) - 1) * np.dot(scores, scores)
    return float(np.dot(weights, beta)), float(np.sqrt(variance))
