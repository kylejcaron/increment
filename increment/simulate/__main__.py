"""CLI entry point: ``uv run python -m increment.simulate``.

Separate from ``runner.py`` to avoid the double-import problem: running a
submodule that the package ``__init__.py`` also imports loads it twice
under different ``sys.modules`` keys, duplicating class identities. This
module is never imported by ``__init__.py``, so it loads only once.
"""

from increment.simulate.runner import main

if __name__ == "__main__":
    main()
