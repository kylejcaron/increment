from pathlib import Path

import pytest
from sybil import Document
from sybil.parsers.markdown import SkipParser

from conftest import OutputBlockParser, PythonOutputBlockParser


def _evaluate(markdown: str, tmp_path: Path) -> None:
    path = tmp_path / "example.md"
    path.write_text(markdown)
    document = Document.parse(
        str(path), PythonOutputBlockParser(), OutputBlockParser(), SkipParser()
    )
    for example in document:
        example.evaluate()


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
    markdown = """??? note \"Example\"
    ```python
    print('folded')
    ```
    <!-- output -->
    ```text
    folded
    ```
"""
    _evaluate(markdown, tmp_path)
