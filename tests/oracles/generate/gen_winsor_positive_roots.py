"""Freeze a positive-outcome bootstrap reference for the pooled winsor method.

The zero-inclusive extension of ``positive-log-kernel-bootstrap-t-v1`` must
preserve the earlier positive-only numerical results within tight tolerances.
This script was run once against the construction that preceded the extension;
commit the JSON it writes and never regenerate it from a later kernel.

Usage (from the repository root):
  uv run python tests/oracles/generate/gen_winsor_positive_roots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_references
from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "winsor_positive_roots.json"


def cases() -> list[dict]:
    rng = np.random.default_rng(9182)
    return [
        {
            "id": "tiny-three-arm-q75",
            "quantile": 0.75,
            "stream": 23,
            "arms": {"C": [1, 2, 3, 9], "T": [2, 3, 5, 8, 12], "O": [1, 4, 7]},
            "treatments": ["T", "O"],
        },
        {
            "id": "lognormal-400x400-q99",
            "quantile": 0.99,
            "stream": 0,
            "arms": {
                "C": rng.lognormal(0.0, 0.5, 400).tolist(),
                "T": rng.lognormal(0.05, 0.5, 400).tolist(),
            },
            "treatments": ["T"],
        },
        {
            "id": "lognormal-300x1200-q95",
            "quantile": 0.95,
            "stream": 0,
            "arms": {
                "C": rng.lognormal(0.0, 1.6, 300).tolist(),
                "T": rng.lognormal(0.05, 1.6, 1200).tolist(),
            },
            "treatments": ["T"],
        },
    ]


def main() -> None:
    records = []
    for case in cases():
        raw = WinsorRawState(
            metric="revenue",
            study_id="oracle",
            missingness="error",
            quantile=case["quantile"],
            inference=WinsorInferenceSpec(stream=case["stream"]),
            arms=tuple(RawArm(group_id=g, values=tuple(v)) for g, v in case["arms"].items()),
        )
        references = full_procedure_bootstrap_references(raw, "C", tuple(case["treatments"]))
        results = {}
        for treatment, reference in references.items():
            results[treatment] = {
                "observed_cutoff": reference.observed_cutoff,
                "pilot_cutoff": reference.pilot_cutoff,
                "log_point": reference.log_relative.point,
                "log_target": reference.log_relative.pilot_target,
                "log_se": reference.log_relative.se,
                "additive_point": reference.additive.point,
                "additive_target": reference.additive.pilot_target,
                "additive_se": reference.additive.se,
                "failure_indices": list(reference.failure_indices),
                "first_log_roots": list(reference.log_relative.roots[:8]),
                "bandwidths": [p.bandwidth for p in reference.pilots],
            }
        records.append({**case, "results": results})
    OUT.write_text(json.dumps({"cases": records}, indent=1) + "\n")


if __name__ == "__main__":
    main()
