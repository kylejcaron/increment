"""Categorical adjustment covariates as fitted level encodings.

An observational adjustment column is numeric or categorical. A categorical
column travels through a cohort as one float column of level codes (NaN
where the level is null), so every stage that masks, slices or imputes rows
keeps working on one matrix. The 0/1 level indicators a learner actually
sees are fitted on that learner's own training rows: the modal training
level is the reference and every other training level gets one column, in
descending frequency with ties broken by label -- the convention
`increment.estimation.cate.DesignSpec` uses. A level the fit never saw
encodes as the reference, exactly as a user-built dummy column that is
identically zero in training would; `UnseenLevels` collects those rows so
the fit stage can disclose them, since a deterministic encoding is no
evidence that the fit has support for that level. A null level reaches a
learner only under ``missing='allow'``, as NaN across the column's level
indicators; a column marked null-carrying keeps at least one indicator
column even when a fit saw a single level, so the NaN is never lost to a
zero-width block. Nothing here reads outcomes, and no level set from
held-out rows shapes a design.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Literal

import numpy as np

from increment.errors import InvalidRequestError, raiser, refusals
from increment.estimation._adjust.learners import Learner

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.adjust_encoding.predict_called": "EncodedLearner.predict called before fit",
    },
)
_raise = raiser(_REFUSALS)

Levels = tuple[str, ...]

_NULL_TYPE_NAMES = frozenset({"NAType", "NaTType"})


def is_null(value: object) -> bool:
    """A null cell as every supported frame backend spells it."""
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or type(value).__name__ in _NULL_TYPE_NAMES
    )


def classify_objects(values: Iterable[object]) -> Literal["numeric", "categorical"] | None:
    """What an object-typed column holds once its nulls are set aside:
    real numbers, strings, or a mixture nothing can adjust on."""
    numeric = categorical = False
    for value in values:
        if is_null(value):
            continue
        if isinstance(value, str):
            categorical = True
        elif isinstance(value, Number | bool | np.bool_) and not isinstance(
            value, complex | np.complexfloating
        ):
            numeric = True
        else:
            return None
        if numeric and categorical:
            return None
    return "categorical" if categorical else "numeric"


def level_codes(values: Iterable[object]) -> tuple[np.ndarray, Levels]:
    """Lexically ordered level labels and each row's code (NaN where null).

    Codes are a lossless relabelling of the strings: assigning them by
    sorted label makes them independent of row order, and a fit that
    orders levels by frequency breaks ties on the code, hence on the label.
    """
    items = list(values)
    levels = tuple(sorted({str(v) for v in items if not is_null(v)}))
    index = {label: float(i) for i, label in enumerate(levels)}
    codes = np.fromiter(
        (math.nan if is_null(v) else index[str(v)] for v in items),
        dtype=float,
        count=len(items),
    )
    return codes, levels


@dataclass(frozen=True, slots=True)
class CovariateLayout:
    """Which columns of an adjustment matrix carry level codes, and their labels.

    ``levels[j]`` is ``None`` for a numeric column and the code-ordered label
    tuple of a categorical one. ``nullable`` names the columns whose null
    cells reach a learner as NaN (``missing='allow'``): a nullable
    categorical keeps at least one level column in every fitted encoding,
    so a null row is never encoded as a zero-width block.
    """

    names: tuple[str, ...]
    levels: tuple[Levels | None, ...]
    nullable: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        assert len(self.names) == len(self.levels), "Covariate names and levels must align"

    @property
    def categorical(self) -> tuple[int, ...]:
        """Column positions holding level codes; empty for a numeric matrix."""
        return tuple(j for j, levels in enumerate(self.levels) if levels is not None)

    def restrict(self, columns: np.ndarray) -> CovariateLayout:
        """The layout of ``X[:, columns]`` for a boolean column mask."""
        kept = np.flatnonzero(columns)
        return CovariateLayout(
            tuple(self.names[j] for j in kept),
            tuple(self.levels[j] for j in kept),
            self.nullable,
        )

    def extended(self, names: Sequence[str]) -> CovariateLayout:
        """This layout with numeric columns appended."""
        return CovariateLayout(
            (*self.names, *names), (*self.levels, *(None for _ in names)), self.nullable
        )

    def with_nullable(self, mask: np.ndarray) -> CovariateLayout:
        """This layout with the flagged columns marked null-carrying."""
        flagged = frozenset(name for name, flag in zip(self.names, mask, strict=True) if flag)
        return CovariateLayout(self.names, self.levels, flagged)

    def _widest(self, j: int) -> int:
        """Columns the ``j``-th covariate can occupy in a fitted design: one
        for a numeric column, one indicator per non-reference level for a
        categorical one, never fewer than one when it is null-carrying."""
        levels = self.levels[j]
        if levels is None:
            return 1
        return max(len(levels) - 1, int(self.names[j] in self.nullable))

    def encoded_width(self) -> int:
        """Widest design a fit on this layout can produce."""
        return sum(self._widest(j) for j in range(len(self.names)))

    def expand_columns(self, mask: np.ndarray) -> np.ndarray:
        """A per-column flag spread over the widest design's columns: a
        flagged categorical flags every one of its level columns."""
        flags: list[bool] = []
        for j in range(len(self.names)):
            flags.extend([bool(mask[j])] * self._widest(j))
        return np.asarray(flags, dtype=bool)


def matrix_from_columns(
    cols: Mapping[str, np.ndarray], names: Sequence[str], n: int
) -> tuple[np.ndarray, CovariateLayout]:
    """One float matrix over *names*: numeric arrays as they are, string
    arrays as level codes. ``n`` sizes the matrix when *names* is empty."""
    columns: list[np.ndarray] = []
    levels: list[Levels | None] = []
    for name in names:
        values = np.asarray(cols[name])
        if values.dtype.kind in "biuf":
            columns.append(values.astype(float, copy=False))
            levels.append(None)
            continue
        codes, labels = level_codes(values.tolist())
        columns.append(codes)
        levels.append(labels)
    layout = CovariateLayout(tuple(names), tuple(levels))
    if not columns:
        return np.empty((n, 0), dtype=float), layout
    return np.column_stack(columns), layout


def modal_code(codes: np.ndarray, observed: np.ndarray) -> float:
    """The most frequent observed code, the lowest (lexically first) on a tie."""
    counts = np.bincount(codes[observed].astype(np.intp))
    return float(int(np.argmax(counts)))


@dataclass(frozen=True, slots=True)
class FittedEncoding:
    """The level indicators one fit's training rows support.

    ``seen[k]`` lists, for the ``k``-th categorical column, the codes
    present in the fitting rows in descending frequency, ties broken by
    code: the first is the reference and every other one gets an indicator
    column. A null-carrying column (``layout.nullable``) whose fit saw
    fewer than two levels keeps one all-zero column, so its null rows still
    reach the learner as NaN.
    """

    layout: CovariateLayout
    seen: tuple[tuple[int, ...], ...]

    @classmethod
    def fit(cls, layout: CovariateLayout, X: np.ndarray) -> FittedEncoding:
        """Learn the reference and indicator set of every categorical column
        from the rows of *X* alone."""
        seen: list[tuple[int, ...]] = []
        for j in layout.categorical:
            codes = X[:, j]
            levels = layout.levels[j]
            assert levels is not None
            observed = codes[~np.isnan(codes)].astype(np.intp)
            counts = np.bincount(observed, minlength=len(levels))
            present = sorted(np.flatnonzero(counts).tolist(), key=lambda c: (-int(counts[c]), c))
            seen.append(tuple(present))
        return cls(layout, tuple(seen))

    def _block(self, k: int, j: int) -> tuple[tuple[int, ...], int]:
        """``(retained codes, block width)`` of the ``k``-th categorical
        column, at layout position ``j``: the non-reference codes each get
        one indicator, and a null-carrying column is never narrower than one."""
        kept = self.seen[k][1:]
        nullable = self.layout.names[j] in self.layout.nullable
        return kept, max(len(kept), int(nullable))

    def width(self) -> int:
        """Columns ``transform`` emits: every numeric column plus every
        categorical column's block."""
        numeric = sum(1 for levels in self.layout.levels if levels is None)
        return numeric + sum(self._block(k, j)[1] for k, j in enumerate(self.layout.categorical))

    def names(self) -> tuple[str, ...]:
        """Design column names in ``transform`` order: ``name`` for a numeric
        column, ``name=level`` for each retained level indicator and
        ``name=<null>`` for a null-carrying column's all-zero block."""
        return tuple(name for name, _ in self._columns())

    def sources(self) -> tuple[str, ...]:
        """The covariate each design column derives from, aligned with ``names``."""
        return tuple(source for _, source in self._columns())

    def _columns(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        k = 0
        for j, (name, levels) in enumerate(zip(self.layout.names, self.layout.levels, strict=True)):
            if levels is None:
                out.append((name, name))
                continue
            kept, width = self._block(k, j)
            k += 1
            out.extend((f"{name}={levels[code]}", name) for code in kept)
            out.extend((f"{name}=<null>", name) for _ in range(width - len(kept)))
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The design of *X* under this fitted encoding: numeric columns in
        place, each categorical column replaced by its retained level
        indicators (all zero for the reference and for any level the fit
        never saw; NaN across the block where the level is null). A layout
        without categorical columns returns *X* itself."""
        if not self.layout.categorical:
            return X
        out = np.empty((X.shape[0], self.width()), dtype=float)
        col = 0
        k = 0
        for j, levels in enumerate(self.layout.levels):
            if levels is None:
                out[:, col] = X[:, j]
                col += 1
                continue
            kept, width = self._block(k, j)
            k += 1
            if not width:
                continue
            codes = X[:, j]
            block = out[:, col : col + width]
            for i, code in enumerate(kept):
                block[:, i] = codes == code
            block[:, len(kept) :] = 0.0
            null = np.isnan(codes)
            if null.any():
                block[null] = math.nan
            col += width
        return out

    def unseen(self, X: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
        """``(covariate, level, rows)`` for every non-null level of *X* this
        fit never saw: the rows ``transform`` encodes as the reference."""
        out: list[tuple[str, str, np.ndarray]] = []
        for k, j in enumerate(self.layout.categorical):
            levels = self.layout.levels[j]
            assert levels is not None
            codes = X[:, j]
            observed = codes[~np.isnan(codes)].astype(np.intp)
            present = np.flatnonzero(np.bincount(observed, minlength=len(levels)))
            for code in present.tolist():
                if code not in self.seen[k]:
                    out.append((self.layout.names[j], levels[code], codes == code))
        return out


class EncodedLearner:
    """A learner whose level encoding is fitted on its own training rows.

    ``fit`` learns the encoding from the rows it is given, then fits the
    wrapped learner on the encoded design; ``predict`` applies that same
    encoding, so a held-out level the fit never saw reads as the reference
    and a null level reaches the wrapped learner as NaN. The wrapped
    learner sees exactly the float design a caller-built dummy encoding
    would have handed it.
    """

    __slots__ = ("encoding", "inner", "layout")

    def __init__(self, inner: Learner, layout: CovariateLayout) -> None:
        self.inner = inner
        self.layout = layout
        self.encoding: FittedEncoding | None = None

    def fit(self, X: np.ndarray, d: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        self.encoding = FittedEncoding.fit(self.layout, X)
        self.inner.fit(self.encoding.transform(X), d)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.inner.predict(self.design(X))

    def design(self, X: np.ndarray) -> np.ndarray:
        """The fitted design of *X*, the matrix ``inner`` was trained on."""
        if self.encoding is None:
            _raise("estimation.adjust_encoding.predict_called")
        return self.encoding.transform(np.asarray(X, dtype=float))

    def restrict(self, columns: np.ndarray) -> EncodedLearner:
        """The same wrapped learner over the column subset ``columns``."""
        return EncodedLearner(self.inner, self.layout.restrict(columns))


def encoded_factory(
    factory: Callable[[], Learner], layout: CovariateLayout
) -> Callable[[], Learner]:
    """*factory* itself for a numeric layout; otherwise a factory whose
    learners fit the level encoding on their own training rows."""
    if not layout.categorical:
        return factory
    return lambda: EncodedLearner(factory(), layout)


def unwrap(learner: Learner) -> Learner:
    """The learner a caller supplied, beneath any encoding."""
    return learner.inner if isinstance(learner, EncodedLearner) else learner


def learner_name(learner: Learner) -> str:
    """The supplied learner's class name, as refusals and notes report it."""
    return type(unwrap(learner)).__name__


def restricted(learner: Learner, columns: np.ndarray) -> Learner:
    """*learner* over the column subset ``columns`` of its matrix."""
    return learner.restrict(columns) if isinstance(learner, EncodedLearner) else learner


def fitted_design(learner: Learner, X: np.ndarray) -> np.ndarray:
    """The matrix *learner* was actually fitted on for the rows of *X*."""
    return learner.design(X) if isinstance(learner, EncodedLearner) else X


class UnseenLevels:
    """Prediction rows whose level no training row of the fit that scored
    them carried, accumulated over one stage's nuisance fits.

    Such a row encodes as that fit's reference level: deterministic, but
    the fit has no support for it, so nothing here is evidence of overlap
    or ignorability. ``record`` notes one fit's prediction rows;
    ``summary`` reports each ``(covariate, level)`` with the distinct rows
    so scored and the number of fits that scored them.
    """

    __slots__ = ("_fits", "_rows", "n")

    def __init__(self, n: int) -> None:
        self.n = n
        self._rows: dict[tuple[str, str], np.ndarray] = {}
        self._fits: dict[tuple[str, str], int] = {}

    def record(
        self, learner: Learner, X: np.ndarray, rows: slice | np.ndarray = slice(None)
    ) -> None:
        """Note the levels of *X* -- the stage's rows *rows* -- that
        *learner*'s fitted encoding never saw; a learner without an
        encoding records nothing."""
        if isinstance(learner, EncodedLearner) and learner.encoding is not None:
            self.record_encoding(learner.encoding, X, rows)

    def record_encoding(
        self, encoding: FittedEncoding, X: np.ndarray, rows: slice | np.ndarray = slice(None)
    ) -> None:
        """Note the levels of *X* -- the stage's rows *rows* -- that
        *encoding* never saw, whichever fit or fixed basis it serves."""
        for name, label, mask in encoding.unseen(np.asarray(X, dtype=float)):
            key = (name, label)
            union = self._rows.get(key)
            if union is None:
                union = self._rows[key] = np.zeros(self.n, dtype=bool)
            union[rows] |= mask
            self._fits[key] = self._fits.get(key, 0) + 1

    def summary(self) -> tuple[tuple[str, str, int, int], ...]:
        """``(covariate, level, n_rows, n_fits)`` per unseen level, ordered
        by covariate then level."""
        return tuple(
            (name, label, int(self._rows[name, label].sum()), self._fits[name, label])
            for name, label in sorted(self._rows)
        )


def unseen_levels_text(summary: Sequence[tuple[str, str, int, int]], n: int) -> str:
    """The disclosure notes and warnings carry for a non-empty
    `UnseenLevels.summary` over *n* scored rows."""
    named = ", ".join(
        f"{covariate}={level} ({n_rows} of {n} rows, {n_fits} fit{'' if n_fits == 1 else 's'})"
        for covariate, level, n_rows, n_fits in summary
    )
    return (
        "categorical level(s) absent from a nuisance fit's training rows were "
        f"scored as that fit's reference level: {named} -- deterministic, but "
        "not evidence of overlap or ignorability for those rows"
    )


def fixed_design(
    layout: CovariateLayout, X: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """The modal-reference design of *X* over its own rows, with column
    names and source covariates: the representation balance diagnostics
    and outside callers without a training boundary read. A numeric layout
    returns *X* itself."""
    if not layout.categorical:
        return X, layout.names, layout.names
    encoding = FittedEncoding.fit(layout, X)
    return encoding.transform(X), encoding.names(), encoding.sources()
