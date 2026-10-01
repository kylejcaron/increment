"""Generate the two fixed samples the discrete quantile-CI oracle compares
against. Run once; commits tests/oracles/data/quantile_values.csv. Both
gen_quantile_oracle.R (R qbinom) and test_quantile_oracle.py
(increment.estimation.quantile.log_quantile_se) read this same file.

Usage (from the repository root):
    uv run python tests/oracles/generate/gen_quantile_samples.py
"""

from __future__ import annotations

import csv

import numpy as np

OUT = "tests/oracles/data/quantile_values.csv"

rng = np.random.default_rng(20260915)
rows: list[dict[str, object]] = []

# n=40, q=0.90: a moderate sample where the Woodruff order-statistic
# bracket sits well inside both tails.
for v in rng.lognormal(mean=1.0, sigma=0.6, size=40):
    rows.append({"case_id": "n40_q90", "q": 0.90, "value": float(v)})

# n=120, q=0.50: a larger sample at the median -- a differently shaped
# binomial bracket (roughly symmetric around n/2) exercising the other
# side of log_quantile_se's rank arithmetic.
for v in rng.lognormal(mean=2.0, sigma=0.4, size=120):
    rows.append({"case_id": "n120_q50", "q": 0.50, "value": float(v)})

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["case_id", "q", "value"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {len(rows)} rows to {OUT}")
