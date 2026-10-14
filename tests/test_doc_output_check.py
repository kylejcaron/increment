from __future__ import annotations

import pytest
from sybil import Document

from conftest import OutputBlockParser, PythonOutputBlockParser


def test_output_marker_checks_invisible_python_block() -> None:
    source = """<!-- invisible-code-block: python
print("captured output")
-->

<!-- output -->
```text
stale output
```
"""
    document = Document(source, "doc.md")
    for parser in (PythonOutputBlockParser(), OutputBlockParser()):
        for region in parser(document):
            document.add(region)

    examples = list(document)
    assert len(examples) == 2
    examples[0].evaluate()
    with pytest.raises(AssertionError, match="documented output does not match"):
        examples[1].evaluate()
