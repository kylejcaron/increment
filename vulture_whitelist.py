"""Vulture false positives (see `vulture increment vulture_whitelist.py
--min-confidence 80`), not dead code: each name below is genuinely read.
"""

_maturity_days
exc_type
tb
RawOutcomeSource  # increment/estimation/winsor.py:59, increment/readouts.py:90 —
# imported only under TYPE_CHECKING and consumed exclusively inside
# cast("RawOutcomeSource", ...), which vulture's static scan can't see.
# Parameter names of ibis builtin SQL function stubs in
# increment/query/artifact_digest.py: ibis reads the signature, not the body.
case
algo
fmt
