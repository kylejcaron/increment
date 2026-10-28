from __future__ import annotations

import re
from pathlib import Path

from tests.refusal_registry import walk_refusal_registries


def test_refusal_guide_covers_registered_code_families() -> None:
    guide = Path(__file__).parents[1] / "docs" / "guides" / "refusals.md"
    documented = set(re.findall(r"`([a-z_]+)\.\*`", guide.read_text()))
    registered = {
        code.split(".", maxsplit=1)[0] for code in walk_refusal_registries() if "." in code
    }

    assert documented == registered


def test_refusal_guide_explains_cross_source_joint_grouping() -> None:
    guide = Path(__file__).parents[1] / "docs" / "guides" / "refusals.md"
    text = guide.read_text()

    assert "`evidence.joint_grouping_unavailable`" in text
    assert "declare all grouping properties together on one source" in text
