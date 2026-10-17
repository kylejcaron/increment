"""Categorical and indicator covariate encoding for multicovariate CUPED."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from increment.estimation._adjust.encoding import is_null
from increment.estimation.cuped._numeric import _fail


class CategoricalMissing(Enum):
    """A pickle-stable categorical null level that cannot equal a user label."""

    VALUE = "typed missing category"


MISSING_CATEGORY = CategoricalMissing.VALUE


@dataclass(frozen=True, slots=True)
class MulticovariateCupedLayout:
    names: tuple[str, ...]
    kinds: tuple[str, ...]
    levels: tuple[tuple[object, ...] | None, ...]
    references: tuple[object, ...]
    encoded_columns: tuple[str, ...]
    structural_omissions: tuple[str, ...]
    shared_indicators: tuple[tuple[str, ...], ...]


def _normalized_offset_difference(offset, reference, scale):
    exponent = max(math.frexp(abs(value))[1] for value in (offset, reference, scale) if value != 0)
    numerator = math.fsum((math.ldexp(offset, -exponent), -math.ldexp(reference, -exponent)))
    denominator = math.ldexp(scale, -exponent)
    if denominator == 0:
        if numerator == 0:
            return 0.0
        _fail(
            "estimation.cuped_linear.nonfinite",
            name="covariate offset",
            row=0,
        )
    result = numerator / denominator
    if not math.isfinite(result):
        _fail(
            "estimation.cuped_linear.nonfinite",
            name="covariate offset",
            row=0,
        )
    return result


def _categorical_values(raw, raw_values, n):
    if isinstance(raw, np.ndarray) and raw.dtype.kind in "US":
        values = raw_values.astype(str) if raw_values.dtype.kind == "S" else raw_values
        return values, np.zeros(n, bool)
    values = np.asarray(raw, dtype=object)
    return values, np.fromiter((is_null(value) for value in values), bool, n)


def _encode_covariates(groups, indices, order, covariates, kinds, missing):
    n = len(groups)
    columns, column_names, transforms = [], [], []
    structural, references, levels = [], [], []
    masks: dict[bytes, tuple[np.ndarray, list[str]]] = {}
    names = tuple(covariates)
    for name, raw in covariates.items():
        if kinds[name] not in ("continuous", "categorical"):
            _fail("estimation.cuped_linear.covariate_kind", name=name, kind=kinds[name])
        raw_values = np.asarray(raw)
        if raw_values.shape != (n,):
            _fail("estimation.cuped_linear.input_shape", name=name, n=n, shape=raw_values.shape)
        if kinds[name] == "categorical":
            values, null = _categorical_values(raw, raw_values, n)
        elif kinds[name] == "continuous" and raw_values.dtype.kind in "biuf":
            values = raw_values
            null = np.isnan(raw_values) if raw_values.dtype.kind == "f" else np.zeros(n, bool)
        else:
            values = np.asarray(raw, dtype=object)
            null = np.fromiter((is_null(value) for value in values), bool, n)
        policy = missing.get(name, "refuse")
        if policy not in ("refuse", "impute-indicator") or (null.any() and policy == "refuse"):
            _fail("estimation.cuped_linear.missing_refused", name=name)
        if kinds[name] == "continuous":
            transform = _encode_numeric(
                name,
                values,
                null,
                indices,
                columns,
                column_names,
                structural,
                references,
                levels,
                masks,
            )
            if transform is not None:
                transforms.append(transform)
        else:
            before = len(columns)
            _encode_category(
                name,
                values,
                null,
                indices,
                order,
                columns,
                column_names,
                structural,
                references,
                levels,
            )
            transforms.extend([((0.0,) * len(order), 1.0)] * (len(columns) - before))
    shared = []
    for indicator, owners in masks.values():
        if np.ptp(indicator) == 0:
            continue
        columns.append(indicator)
        column_names.append(f"missing[{','.join(owners)}]")
        transforms.append(((0.0,) * len(order), 1.0))
        shared.append(tuple(owners))
    matrix = np.empty((n, len(columns)), dtype=float)
    j = 0
    while j < len(columns):
        column = columns[j]
        if isinstance(column, tuple):
            codes = column[0]
            stop = j + 1
            while stop < len(columns) and isinstance(columns[stop], tuple):
                if columns[stop][0] is not codes:
                    break
                stop += 1
            selected = np.fromiter(
                (columns[i][1] for i in range(j, stop)), dtype=int, count=stop - j
            )
            np.equal(codes[:, None], selected[None, :], out=matrix[:, j:stop])
            j = stop
        else:
            matrix[:, j] = column
            j += 1
    x = matrix
    encoded_names = tuple(column_names)
    active_transforms = tuple(transforms)
    layout = MulticovariateCupedLayout(
        names,
        tuple(kinds[name] for name in names),
        tuple(levels),
        tuple(references),
        encoded_names,
        tuple(structural),
        tuple(shared),
    )
    return (
        x,
        layout,
        tuple(item[0] for item in active_transforms),
        tuple(item[1] for item in active_transforms),
    )


def _encode_numeric(
    name, values, null, indices, columns, column_names, structural, references, levels, masks
):
    if values.dtype.kind in "biuf":
        numeric = np.asarray(values, dtype=float)
        if null.any():
            numeric = np.where(null, 0.0, numeric)
    else:
        numeric = np.zeros(len(values), dtype=float)
        for i, value in enumerate(values):
            if not null[i]:
                try:
                    numeric[i] = float(value)
                except (TypeError, ValueError, OverflowError):
                    _fail(
                        "estimation.cuped_linear.covariate_kind",
                        name=name,
                        kind=type(value).__name__,
                    )
    if not np.isfinite(numeric).all():
        _fail(
            "estimation.cuped_linear.nonfinite",
            name=name,
            row=int(np.flatnonzero(~np.isfinite(numeric))[0]),
        )
    levels.append(None)
    references.append(0.0)
    if null.any():
        key = null.tobytes()
        if key in masks:
            masks[key][1].append(name)
        else:
            masks[key] = (null.astype(float), [name])
    offsets = []
    centered = np.empty_like(numeric)
    for group in indices:
        present = numeric[group[~null[group]]]
        offset = float(present[0]) if len(present) else 0.0
        offsets.append(offset)
        centered[group] = numeric[group] - offset
    if not np.isfinite(centered).all():
        offsets = [0.0] * len(indices)
        centered = numeric.copy()
    scale = float(np.max(np.abs(centered), initial=0.0))
    if scale == 0:
        if np.ptp(numeric) == 0:
            structural.append(name)
            return None
        columns.append(centered)
        column_names.append(name)
        return tuple(offsets), 1.0
    columns.append(centered / scale)
    column_names.append(name)
    return tuple(offsets), scale


def _encode_category(
    name, values, null, indices, order, columns, column_names, structural, references, levels
):
    labels_values = values[~null]
    if values.dtype.kind not in "US" and any(not isinstance(value, str) for value in labels_values):
        _fail(
            "estimation.cuped_linear.covariate_kind",
            name=name,
            kind="categorical values must be strings",
        )
    labels_array = np.asarray(labels_values, dtype=str)
    sorted_labels, codes = np.unique(labels_array, return_inverse=True)
    labels = tuple(str(label) for label in sorted_labels)
    full_levels: list[object] = list(labels)
    if null.any():
        full_levels.append(MISSING_CATEGORY)
    codes_full = np.full(len(values), -1, dtype=int)
    codes_full[~null] = codes.astype(int)
    if null.any():
        codes_full[null] = len(labels)
    ref_code = int(np.argmax(np.bincount(codes_full, minlength=len(full_levels))))
    references.append(full_levels[ref_code])
    levels.append(tuple(labels) + ((MISSING_CATEGORY,) if null.any() else ()))
    if len(full_levels) <= 1:
        structural.append(name)
        return
    arm_codes = np.empty(len(values), dtype=int)
    for arm_code, idx in enumerate(indices):
        arm_codes[idx] = arm_code
    per_arm = np.bincount(
        codes_full * len(order) + arm_codes,
        minlength=len(full_levels) * len(order),
    ).reshape(len(full_levels), len(order))
    deficient = np.flatnonzero(np.min(per_arm, axis=1) < 2)
    if len(deficient):
        code = int(deficient[0])
        level = full_levels[code]
        shown = "<typed-missing>" if level is MISSING_CATEGORY else level
        _fail(
            "estimation.cuped_linear.level_support",
            column=name,
            level=shown,
            counts=dict(zip(order, per_arm[code], strict=True)),
        )
    for code, level in enumerate(full_levels):
        if code == ref_code:
            continue
        shown = "<typed-missing>" if level is MISSING_CATEGORY else repr(level)
        columns.append((codes_full, code))
        column_names.append(f"{name}[{shown}]")
