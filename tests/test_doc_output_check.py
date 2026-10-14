from __future__ import annotations

from pathlib import Path
from unittest import SkipTest

import pytest
from sybil import Document
from sybil.parsers.markdown import SkipParser

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


def _evaluate(markdown: str, tmp_path: Path) -> None:
    path = tmp_path / "example.md"
    path.write_text(markdown)
    document = Document.parse(
        str(path), PythonOutputBlockParser(), OutputBlockParser(), SkipParser()
    )
    for example in document:
        try:
            example.evaluate()
        except SkipTest:
            pass


def test_output_block_must_match_captured_python_output(tmp_path):
    markdown = """```python
print(1.5)
```
<!-- output -->
```text
1.6
```
"""
    with pytest.raises(AssertionError, match="1.5"):
        _evaluate(markdown, tmp_path)


def test_matching_output_ignores_unmarked_text_and_normalizes_whitespace(tmp_path):
    markdown = """```python
print('1.5   units')
```
```text
not checked
```
<!-- output -->
```text
1.5 units
```
"""
    _evaluate(markdown, tmp_path)


def test_output_block_after_reason_bearing_skip_is_exempt(tmp_path):
    markdown = """<!-- skip: next "requires a prepared frame" -->
```python
print('not executed')
```
<!-- output -->
```text
anything
```
"""
    _evaluate(markdown, tmp_path)


def test_false_conditional_next_still_checks_output(tmp_path):
    markdown = """<!-- skip: next if (False) -->
```python
print(1.5)
```
<!-- output -->
```text
1.6
```
"""
    with pytest.raises(AssertionError, match="1.5"):
        _evaluate(markdown, tmp_path)


def test_false_conditional_skip_range_still_checks_output(tmp_path):
    markdown = """<!-- skip: start if (False) -->
```python
print(1.5)
```
<!-- output -->
```text
1.6
```
<!-- skip: end -->
"""
    with pytest.raises(AssertionError, match="1.5"):
        _evaluate(markdown, tmp_path)


def test_true_conditional_skip_exempts_output(tmp_path):
    markdown = """<!-- skip: next if (True) -->
```python
print('not executed')
```
<!-- output -->
```text
anything
```
"""
    _evaluate(markdown, tmp_path)


def test_output_block_after_skipped_python_is_exempt(tmp_path):
    markdown = """<!-- skip: next -->
```python
print('not executed')
```
<!-- output -->
```text
anything
```
"""
    _evaluate(markdown, tmp_path)


def test_output_checker_finds_fences_inside_admonition_fold(tmp_path):
    markdown = """??? note "Example"
    ```python
    print('folded')
    ```
    <!-- output -->
    ```text
    folded
    ```
"""
    _evaluate(markdown, tmp_path)
