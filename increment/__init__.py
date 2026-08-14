"""A/B testing querying and stats engine.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("increment")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
