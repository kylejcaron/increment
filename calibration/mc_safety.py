"""Run the full Monte Carlo grids removed from pytest's slow tier.

Invoke with ``python -m calibration.mc_safety <campaign> --out results.jsonl``.
Use ``--smoke`` for a small diagnostic execution; smoke records are not calibration
acceptance evidence. Full runs call the same assertions as their former pytest grids
and append one JSON result record per declared cell.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import asdict
from pathlib import Path


def _record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _c07(out: Path, *, smoke: bool) -> int:
    from tests.test_cate_calibration import (
        _c07_calibration_report,
        _c07_null_validation,
        _c07_record_calibration,
        test_c07_cluster_null_rejection_and_coverage,
    )

    if smoke:
        result = _c07_null_validation(10, 0.2, "equal", False, False, 0, repetitions=9, alpha=0.2)
        ledger = {}
        _c07_record_calibration(ledger, result)
        _record(
            out,
            {
                "campaign": "c07",
                "scope": "diagnostic_smoke",
                "report": _c07_calibration_report(ledger),
            },
        )
        return 0
    cells = [
        (k, icc, weighting, imbalanced, stress)
        for k in (10, 40, 200)
        for icc in (0.0, 0.2, 0.5)
        for weighting in ("member_count", "equal")
        for imbalanced in (False, True)
        for stress in (False, True)
    ]
    for values in cells:
        props = {}
        test_c07_cluster_null_rejection_and_coverage(*values, props.__setitem__)
        _record(
            out,
            {
                "campaign": "c07",
                "scope": "full_grid",
                "cell": values,
                "acceptance": json.loads(props["c07_calibration"]),
            },
        )
    return 0


def _quantile(out: Path, *, smoke: bool) -> int:
    from tests.estimation.test_quantile_recovery import _null_rejection_rate, mcse

    if smoke:
        sims, n, kind = 20, 5000, "ms"
        rate = _null_rejection_rate(kind, n, 0.5, 0.05, sims)
        ceiling = 0.05 + 4 * mcse(0.05, sims)
        assert rate <= ceiling
        _record(
            out,
            {
                "campaign": "quantile_lattice",
                "scope": "diagnostic_smoke",
                "cell": [kind, n, sims],
                "rejection_rate": rate,
                "ceiling": ceiling,
            },
        )
        return 0
    for kind, n, sims in (
        ("seconds", 30000, 20000),
        ("ms", 10000, 6000),
        ("cents", 20000, 6000),
    ):
        rate = _null_rejection_rate(kind, n, 0.5, 0.05, sims)
        ceiling = 0.05 + 4 * mcse(0.05, sims)
        assert rate <= ceiling, f"{kind} n={n}: null rejection {rate} > {ceiling} ({sims} sims)"
        _record(
            out,
            {
                "campaign": "quantile_lattice",
                "scope": "full_grid",
                "cell": [kind, n, sims],
                "rejection_rate": rate,
                "ceiling": ceiling,
            },
        )
    return 0


def _conversion(out: Path, *, smoke: bool) -> int:
    from calibration import conversion_route as cr

    if smoke:
        cell = cr.cells(cr.dense_min_count(0.1) - 1)[0]
        result = cr.simulate_hybrid(
            cell, alpha=0.2, alternative="two-sided", reps=10, seed=20261004
        )
        _record(
            out,
            {
                "campaign": "conversion_hybrid",
                "scope": "diagnostic_smoke",
                "cell": repr(cell),
                "replications": 10,
                "result": asdict(result),
            },
        )
        return 0
    for alternative in ("two-sided", "greater", "less"):
        passed = cr.replicate_check(0.1, workers=1, alternative=alternative)
        assert passed
        _record(
            out,
            {
                "campaign": "conversion_hybrid",
                "scope": "full_grid",
                "alternative": alternative,
                "acceptance": passed,
            },
        )
    return 0


class _Diagnostics:
    @contextlib.contextmanager
    def stage(self, _name):
        yield

    def progress(self, **_kwargs):
        pass


def _switchback(out: Path, *, smoke: bool) -> int:
    from tests._switchback_planning_design import CELLS
    from tests.power.test_switchback_planning_calibration import (
        test_source_estimator_and_planning_variance_calibration,
    )

    if smoke:
        from tests.power.test_switchback_planning_calibration import (
            _capture_source,
            _panel,
            _streams,
        )

        cell = next(cell for cell in CELLS if cell.id == "unit-p0.75-c20-o4-k0")
        frame, _ = _panel(cell, 1, _streams(2026101701, CELLS.index(cell)))
        source, captured = _capture_source(frame, cell)
        _record(
            out,
            {
                "campaign": "switchback_planning",
                "scope": "diagnostic_smoke",
                "cell": cell.id,
                "metrics": sorted(captured),
                "source_metrics": [metric.name for metric in source.metrics],
            },
        )
        return 0
    for index, cell in enumerate(CELLS):
        properties = {}
        test_source_estimator_and_planning_variance_calibration(
            index, cell, properties.__setitem__, _Diagnostics()
        )
        _record(
            out,
            {
                "campaign": "switchback_planning",
                "scope": "full_grid",
                "cell": cell.id,
                "results": properties,
            },
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "campaign", choices=("c07", "quantile-lattice", "conversion-hybrid", "switchback")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    runner = {
        "c07": _c07,
        "quantile-lattice": _quantile,
        "conversion-hybrid": _conversion,
        "switchback": _switchback,
    }[args.campaign]
    return runner(args.out, smoke=args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
