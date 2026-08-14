"""A/B testing querying and stats engine.

This alpha ships one piece of the eventual engine: the conjugate
Normal-Normal update that every lift computation is built on. Import it
from its permanent home rather than the top level::

    from increment.estimation.inference import Normal

The curated top-level API (``Analysis``, ``LiftEstimate``, the power
solvers, and the rest) is deliberately not re-exported here yet. Adding
names to ``__all__`` now and removing them later would be a breaking
change in a subsequent alpha; this module stays minimal until the full
public surface lands.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("increment")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
