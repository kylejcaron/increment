"""Array protocol for caller-supplied targeting scores."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class ScoreDesign(np.ndarray):
    """The float design a caller-supplied score reads, with its column names.

    Numeric adjustment columns pass through as they are; each categorical
    column is replaced by the 0/1 level indicators of one basis fitted on
    the workflow's training rows alone -- the modal training level is the
    reference, every other training level gets one column in descending
    training frequency, and a scored level the training rows never carried
    reads as the reference. One workflow hands every call the same basis,
    so a level means the same columns in each of them. ``columns`` names
    the columns (``name`` for a numeric column, ``name=level`` for an
    indicator) and ``sources`` the adjustment covariate each derives from.
    Row-only indexing and copies retain the names. Other views clear them,
    since a changed axis cannot inherit the original column meanings.
    """

    columns: tuple[str, ...]
    sources: tuple[str, ...]

    def __new__(
        cls, matrix: np.ndarray, columns: tuple[str, ...], sources: tuple[str, ...]
    ) -> ScoreDesign:
        design = np.asarray(matrix, dtype=float).view(type=cls)
        design.columns = columns
        design.sources = sources
        return design

    def __array_finalize__(self, _obj: object) -> None:
        self.columns, self.sources = (), ()

    def __getitem__(self, key: Any) -> Any:
        result = super().__getitem__(key)
        parts = key if isinstance(key, tuple) else (key,)
        rows_only = self.ndim == 2 and (
            len(parts) == 1
            or (
                len(parts) == 2
                and (
                    parts[1] is Ellipsis
                    or (
                        isinstance(parts[1], slice)
                        and parts[1].indices(self.shape[1]) == (0, self.shape[1], 1)
                    )
                )
            )
        )
        if isinstance(result, ScoreDesign) and result.ndim == 2 and rows_only:
            result.columns, result.sources = self.columns, self.sources
        return result

    def copy(self, order: Any = "C") -> ScoreDesign:
        return ScoreDesign(self.view(np.ndarray).copy(order=order), self.columns, self.sources)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        out: tuple[object, ...] | None = None,
        **kwargs: object,
    ) -> object:
        plain = [x.view(np.ndarray) if isinstance(x, ScoreDesign) else x for x in inputs]
        if out is not None:
            kwargs["out"] = tuple(
                x.view(np.ndarray) if isinstance(x, ScoreDesign) else x for x in out
            )
        return getattr(ufunc, method)(*plain, **kwargs)


ArrayPsiFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
    tuple[np.ndarray, np.ndarray],
]

PsiFn = Callable[
    [np.ndarray, np.ndarray, ScoreDesign, np.ndarray, np.ndarray | None],
    tuple[np.ndarray, np.ndarray],
]
