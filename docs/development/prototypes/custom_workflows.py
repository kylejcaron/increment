"""Two private authoring interfaces on one immutable, closed-world orders panel."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import wraps
from itertools import product

import narwhals as nw
import numpy as np
import pyarrow as pa
from scipy.stats import mannwhitneyu

import increment as inc
from increment.breakout.estimates import DailyLiftEstimates, LiftEstimates

DAYS = tuple(date(2026, 1, d) for d in (1, 2, 3, 4))
CONTROL, TREATMENTS = "C", ("B", "A")
ARMS = (CONTROL, *TREATMENTS)
TOTAL, DAILY, ASOF = inc.View.overall(), inc.View.daily(), inc.View.asof()
# ID, assigned arm, pre-exposure baseline, first eligible day, dated orders.
# An omitted event after entry means zero. Jan 4 is beyond the observation end.
UNITS = (
    ("c0x", "C", 0, 1, {1: 1, 3: 3}),
    ("c0y", "C", 0, 1, {1: 2, 2: 0}),
    ("c2x", "C", 2, 2, {1: 99, 2: 2}),
    ("c2y", "C", 2, 2, {2: 2}),
    ("a1x", "A", 1, 1, {1: 1, 2: 1}),
    ("a1y", "A", 1, 1, {1: 2, 2: 1}),
    ("a3x", "A", 3, 2, {2: 3, 3: 1}),
    ("a3y", "A", 3, 2, {2: 3}),
    ("b2x", "B", 2, 2, {1: 99, 3: 3}),
    ("b2y", "B", 2, 2, {2: 0}),
    ("b4x", "B", 4, 2, {2: 8}),
    ("b4y", "B", 4, 2, {2: 8, 4: 999}),
    ("future", "B", 6, 4, {4: 999}),
)


def oracle(view, day):
    end = 3 if day is None else DAYS.index(day) + 1
    return {
        unit: (
            arm,
            pre,
            sum(
                y for d, y in events.items() if entry <= d <= end and (view != "daily" or d == end)
            ),
        )
        for unit, arm, pre, entry, events in UNITS
        if entry <= end
    }


@dataclass(frozen=True)
class Slice:
    view: str
    day: date | None
    units: pa.Table
    counts: dict
    control: str = CONTROL
    treatments: tuple = TREATMENTS

    def key(self, arm):
        return self.view, self.day, arm


def prepared_slices(experiment, reader, views):
    """Private adapter: unwindowed assigned orders only; full, unsegmented calendar views."""
    from increment._frame_panel import _day_labels, _day_population, _with_day_index

    if not set(views) <= set(experiment.views) & {TOTAL, DAILY, ASOF}:
        raise ValueError("Only the declared overall, daily and as-of orders views are supported")
    source = reader._source
    (metric,) = source.context.metrics

    def cell(view, day, units):
        return Slice(view, day, units, Counter(units["group_id"].to_pylist()))

    if TOTAL in views:
        yield cell("total", None, source.unit_frame(metric, covariates=("pre",)))
    if DAILY not in views and ASOF not in views:
        return
    panel = source._observed_days_panel()
    ordinal = source._identity_ordinal
    roster = source._day_identity().join(source._resolve_panel_covariate("pre"), on="unit_id")
    prefix = np.zeros(len(roster)) if ASOF in views else None
    for day in _day_labels(panel):  # every in-scope calendar day occurs on this fixture
        rows = _day_population(
            panel, identity=roster, ds=day, value_columns=["orders"], ordinal=ordinal
        )
        rows, age = _with_day_index(rows, source._declared_exposure())
        eligible = rows.filter(nw.col(age) >= 0).sort(ordinal)
        units = eligible.select(
            "unit_id", "group_id", "pre", nw.col("orders").alias("y")
        ).to_arrow()
        if DAILY in views:
            yield cell("daily", day, units)
        if prefix is not None:
            index = eligible[ordinal].to_numpy()
            prefix[index] += units["y"].to_numpy()
            yield cell("asof", day, units.set_column(3, "y", pa.array(prefix[index])))


def rank(cell):
    group = np.array(cell.units["group_id"].to_pylist())
    y = cell.units["y"].to_numpy()
    c = y[group == cell.control]
    result = {}
    for arm in cell.treatments:
        t = y[group == arm]
        if not len(t) or not len(c):
            result[arm] = {"rank": None, "nominal_p": None, "rank_reason": "missing arm"}
        else:
            u, p = mannwhitneyu(
                t, c, alternative="two-sided", method="asymptotic", use_continuity=True
            )
            result[arm] = {
                "rank": float(u / (len(t) * len(c))),
                "nominal_p": float(p),
                "rank_reason": None,
            }
    return result


def adjusted(cell):
    """One common within-arm least-squares slope; no standard error or test claimed."""
    group = np.array(cell.units["group_id"].to_pylist())
    x, y = (cell.units[col].to_numpy() for col in ("pre", "y"))
    xx = xy = 0.0
    means = {}
    for arm in (cell.control, *cell.treatments):
        mask = group == arm
        if mask.any():
            xa, ya = x[mask], y[mask]
            mx, my = means[arm] = (xa.mean(), ya.mean())
            dx = xa - mx
            xx += dx @ dx
            xy += dx @ (ya - my)
    result = {}
    for arm in cell.treatments:
        if arm not in means or cell.control not in means:
            point, reason = None, "missing arm"
        elif not xx:
            point, reason = None, "no within-arm baseline variance"
        else:
            xc, yc = means[cell.control]
            xt, yt = means[arm]
            point, reason = float((yt - yc) - (xy / xx) * (xt - xc)), None
        result[arm] = {"adjusted": point, "adjusted_reason": reason}
    return result


@dataclass
class Measurements:
    preparations: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    joins: Counter = field(default_factory=Counter)

    def prepare(self, exp, reader, views):
        for s in prepared_slices(exp, reader, views):
            rows = s.units.to_pylist()
            actual = {r["unit_id"]: (r["group_id"], r["pre"], r["y"]) for r in rows}
            assert len(actual) == len(rows) and actual == oracle(s.view, s.day)
            self.preparations.append((s.view, s.day))
            yield s

    def measure(self, fn):
        @wraps(fn)
        def call(s):
            self.calls.append((s.view, s.day, fn.__name__, (s.control, *s.treatments)))
            result = fn(s)
            assert tuple(result) == s.treatments
            return result

        return call

    def align(self, owner, standard, key):
        self.joins[owner] += 1
        return standard[key]


def built_in(exp, reader, views):
    run = inc.run(exp, reader, views=views)
    rows = []
    for view in views:
        collection = run.readout(view)
        assert isinstance(collection, (LiftEstimates, DailyLiftEstimates))
        rows.extend(((view.grain, getattr(r, "ds", None), r.group_id), r) for r in collection)
    assert len(rows) == len(dict(rows))
    return dict(rows)


def ordinary(exp, reader, views, m, rank_fn, adjusted_fn):
    """Analyst call site: use ordinary functions, then align both outputs by label."""
    standard = built_in(exp, reader, views)
    joined = []
    for s in m.prepare(exp, reader, views):
        wins = rank_fn(s)
        differences = adjusted_fn(s)
        for arm in s.treatments:
            key = s.key(arm)
            joined.append(
                (
                    key,
                    m.align("analyst", standard, key),
                    {"rank": wins[arm], "adjusted": differences[arm]},
                    s.counts[s.control],
                    s.counts[arm],
                )
            )
    return joined


@dataclass(frozen=True)
class Method:
    name: str
    evaluate: Callable


def run_with_comparisons(exp, reader, views, m, *, methods):
    """Local wrapper, not increment.run's capture or decision machinery."""
    names = tuple(method.name for method in methods)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate method name")
    standard = built_in(exp, reader, views)
    joined = []
    for s in m.prepare(exp, reader, views):
        outputs = {method.name: method.evaluate(s) for method in methods}
        for arm in s.treatments:
            key = s.key(arm)
            values = {name: output[arm] for name, output in outputs.items()}
            joined.append(
                (key, m.align("runner", standard, key), values, s.counts[s.control], s.counts[arm])
            )
    return joined


def check(joined, views, m):
    expected = [
        (v.grain, day, arm)
        for v in views
        for day in ((None,) if v == TOTAL else DAYS[:3])
        for arm in TREATMENTS
    ]
    keys = Counter(row[0] for row in joined)
    missing = len(set(expected) - keys.keys())
    duplicate = sum(keys.values()) - len(keys)
    assert not missing and not duplicate and set(expected) == keys.keys()
    slices = {(view, day) for view, day, _ in expected}
    assert Counter(m.preparations) == Counter(slices)
    assert Counter(m.calls) == Counter(
        (v, d, name, ARMS) for v, d in slices for name in ("rank", "adjusted")
    )
    cells, mean_checks = {}, 0
    for key, builtin, by_method, nc, nt in joined:
        values = by_method["rank"] | by_method["adjusted"]  # Fixture-specific reporting schema.
        view, day, arm = key
        data = list(oracle(view, day).values())
        yc, yt = ([y for a, _, y in data if a == group] for group in (CONTROL, arm))
        assert (nc, nt) == (len(yc), len(yt))
        if nt:
            assert (builtin.n_control, builtin.n_treat) == (nc, nt)
            np.testing.assert_allclose(
                [builtin.mean_control, builtin.mean_treat], [np.mean(yc), np.mean(yt)]
            )
            mean_checks += 1
            u = sum((t > c) + 0.5 * (t == c) for t, c in product(yt, yc))
            assert values["rank"] == u / (nc * nt) and 0 <= values["nominal_p"] <= 1
            groups = sorted({a for a, _, _ in data})
            design = np.array([[*(a == g for g in groups), x] for a, x, _ in data])
            coef, _, dimension, _ = np.linalg.lstsq(design, [y for _, _, y in data], rcond=None)
            if dimension == design.shape[1]:
                ref = coef[groups.index(arm)] - coef[groups.index(CONTROL)]
                np.testing.assert_allclose(values["adjusted"], ref, atol=1e-12)
            else:
                assert (
                    values["adjusted"] is None
                    and values["adjusted_reason"] == "no within-arm baseline variance"
                )
        else:
            assert builtin.unavailable == "no_treatment_arm"
            assert values["rank"] is values["adjusted"] is values["nominal_p"] is None
            assert values["rank_reason"] == values["adjusted_reason"] == "missing arm"
        cells[key] = dict(
            view=view,
            day=day.isoformat() if day else None,
            treatment=arm,
            n_control=nc,
            n_treatment=nt,
            **values,
        )
    return [cells[k] for k in expected], {
        "slice preparations": len(m.preparations),
        "distinct preparations": len(set(m.preparations)),
        "kernel invocations": dict(Counter(name for _, _, name, _ in m.calls)),
        "ordered roster on every invocation": list(ARMS),
        "missing keys": missing,
        "duplicate keys": duplicate,
        "analyst-owned alignments": m.joins["analyst"],
        "runner-owned alignments": m.joins["runner"],
        "built-in count and mean pairs matched": mean_checks,
    }


def compare(exp, reader, views):
    a, b = Measurements(), Measurements()
    ordinary_rows = ordinary(exp, reader, views, a, a.measure(rank), a.measure(adjusted))
    # Analyst call site: select both explicit method objects; no result-key join at the call site.
    managed_rows = run_with_comparisons(
        exp,
        reader,
        views,
        b,
        methods=(Method("rank", b.measure(rank)), Method("adjusted", b.measure(adjusted))),
    )
    ordinary_cells, ordinary_counts = check(ordinary_rows, views, a)
    managed_cells, managed_counts = check(managed_rows, views, b)
    assert ordinary_cells == managed_cells
    return ordinary_cells, {
        "views": [v.grain for v in views],
        "keys and values agree": True,
        "ordinary functions": ordinary_counts,
        "explicit methods": managed_counts,
    }


def main():
    panel = pa.Table.from_pylist(
        [
            {"id": u, "arm": a, "pre": x, "entry": DAYS[e - 1], "day": DAYS[d - 1], "orders": y}
            for u, a, x, e, events in UNITS
            for d, y in events.items()
        ]
    )
    design = inc.Randomized(
        control_group=CONTROL,
        allocation=dict.fromkeys(ARMS, 1 / 3),
        allocation_scheme="independent",
    )
    exp, panel_reader = inc.from_frame(
        panel,
        shape="panel",
        date="day",
        exposure_date="entry",
        observation_end=DAYS[2],
        unit="id",
        group="arm",
        metrics={"orders": "mean"},
        design=design,
    )
    exp = exp.with_views(DAILY, ASOF)
    with panel_reader:
        cells, original = compare(exp, panel_reader, (TOTAL, DAILY, ASOF))
        changed_cells, changed = compare(exp, panel_reader, (ASOF,))
    assert changed_cells == [c for c in cells if c["view"] == "asof"]
    summary = pa.Table.from_pylist(
        [
            {"id": u, "arm": a, "pre": x, "orders": y}
            for u, (a, x, y) in oracle("total", None).items()
        ]
    )
    exp, reader = inc.from_frame(
        summary, shape="summary", unit="id", group="arm", metrics={"orders": "mean"}, design=design
    )
    with reader:
        totals, summary_counts = compare(exp, reader, (TOTAL,))
    assert totals == [c for c in cells if c["view"] == "total"]
    pooled = [
        c["adjusted"] for c in cells if c["view"] == "daily" and c["day"] == DAYS[1].isoformat()
    ]
    assert pooled == [-1.0, -1.0]  # Pairwise fits here would give -2 for B and 0 for A.
    print(
        json.dumps(
            {
                "cells": cells,
                "panel": [original, changed],
                "matched summary": summary_counts,
                "summary matches panel total": True,
                "unchanged as-of cells after selection change": len(changed_cells),
                "eligibility": "Assigned through the date, inclusive. Missing event = zero; before entry and after Jan 3 excluded. No skip policy or window.",
                "rank": "P(T>C) + 0.5 P(T=C); nominal two-sided asymptotic U test of equal distributions, tie and continuity corrected.",
                "adjusted": "Absolute mean orders difference at a common baseline, one slope fitted across all present arms per slice. Estimate only.",
                "interpretation": "Both paths prepare once per slice, call each kernel once, and produce the same cells. The wrapper moves labelled joins from analyst to runner; it does not remove the joins or a data read.",
                "limitations": "Engineer demonstration on a fixed Arrow frame, not a usability trial. Built-ins run first; custom rows read the same reader separately, not its execution capture. No warehouse, sequential, decision-family, or general calendar integration.",
            },
            allow_nan=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
