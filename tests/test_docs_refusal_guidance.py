from __future__ import annotations

import re
from pathlib import Path

from tests.test_refusal_uniqueness import _walk_refusal_registries


def test_refusal_guide_covers_registered_code_families() -> None:
    guide = Path(__file__).parents[1] / "docs" / "guides" / "refusals.md"
    documented = set(re.findall(r"`([a-z_]+)\.\*`", guide.read_text()))
    registered = {
        code.split(".", maxsplit=1)[0] for code in _walk_refusal_registries() if "." in code
    }

    assert documented == registered
