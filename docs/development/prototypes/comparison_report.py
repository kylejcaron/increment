"""Run the private probes and built-in family controls; emit one JSON report."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import polars as pl

import increment as inc
from increment.breakout.estimates import LiftEstimates


def registered_examples():
    """Independent prefix runs, not continuation or a custom adapter in the production engine."""
    rng = random.Random(37)
    arms = [rng.choice(["C", "T"]) for _ in range(96)]
    metrics = ("goal", "quality", "repeat")
    frame = pl.DataFrame(
        {
            "id": [f"u{i:03d}" for i in range(len(arms))],
            "order": range(len(arms)),
            "arm": arms,
            **{
                name: [
                    int(rng.random() < (0.84 if arm == "T" and name == "quality" else 0.42))
                    for arm in arms
                ]
                for name in metrics
            },
        }
    )
    design = inc.Randomized(
        control_group="C", allocation={"C": 0.5, "T": 0.5}, allocation_scheme="independent"
    )
    out = []
    for kind in ("fixed", "always_valid"):
        options = (
            {"inference": inc.InferenceSpec(kind=kind, baseline_rate=0.42)}
            if kind == "always_valid"
            else {}
        )
        plan = inc.AnalysisPlan(
            primary="goal", secondaries=["quality", "repeat"], q=0.10, **options
        )
        for n in (48, 96):
            experiment, reader = inc.from_frame(
                frame.head(n),
                shape="summary",
                unit="id",
                group="arm",
                design=design,
                exposure_date="order",
                experiment_id="prototype-joint",
                plan=plan,
                metrics=[inc.MetricSpec(name=m, type="conversion") for m in metrics],
            )
            try:
                run = inc.run(experiment, reader)
                rows = run.readout()
                assert isinstance(rows, LiftEstimates)
                roster = [row for row in rows if row.role == "secondary"]
                assert {r.metric for r in roster} == {"quality", "repeat"}
                assert len({r.family_id for r in roster}) == 1
                procedure = "e_bh" if kind == "always_valid" else "bh"
                assert all(r.family_procedure == procedure and r.family_size == 2 for r in roster)
                if kind == "always_valid":
                    for row in rows:
                        cp = row.require_sequential_result().checkpoint
                        assert cp.control.n + cp.treatment.n == n
                out.append(
                    {
                        "kind": kind,
                        "n": n,
                        "procedure": procedure,
                        "family": [r.metric for r in roster],
                        "selected": [r.metric for r in roster if r.discovery],
                        "verdict": run.decisions.program.status,
                    }
                )
            finally:
                reader.close()
    return out


def probe(name):
    script = Path(__file__).with_name(name)
    return json.loads(subprocess.check_output([sys.executable, str(script)], text=True))


def main():
    data = {
        "authoring": probe("custom_workflows.py"),
        "inference": probe("custom_evidence.py"),
        "registered": registered_examples(),
    }
    print(json.dumps(data, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
