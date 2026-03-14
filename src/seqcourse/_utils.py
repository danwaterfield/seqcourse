from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def ensure_object_matrix(data: object) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if isinstance(data, pd.DataFrame):
        values = data.to_numpy(dtype=object)
        row_labels = tuple(str(index) for index in data.index)
        column_labels = tuple(str(column) for column in data.columns)
        return values, row_labels, column_labels
    array = np.asarray(data, dtype=object)
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional array-like input.")
    row_labels = tuple(str(index) for index in range(array.shape[0]))
    column_labels = tuple(str(index) for index in range(array.shape[1]))
    return array, row_labels, column_labels


def normalize_method(method: str) -> str:
    return method.strip().upper()


def stable_unique(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return tuple(ordered)


def entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def normalize_weights(weights: Sequence[float] | None, size: int) -> np.ndarray:
    if weights is None:
        return np.ones(size, dtype=float)
    array = np.asarray(weights, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"Expected {size} weights, received shape {array.shape}.")
    return array


def condensed_from_square(matrix: np.ndarray) -> np.ndarray:
    n_rows = matrix.shape[0]
    result = np.empty(n_rows * (n_rows - 1) // 2, dtype=float)
    index = 0
    for row in range(n_rows - 1):
        width = n_rows - row - 1
        result[index : index + width] = matrix[row, row + 1 :]
        index += width
    return result


def weighted_proportions(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts / total


def trailing_length(row: np.ndarray, void_code: int = 0) -> int:
    non_void = np.flatnonzero(row != void_code)
    if non_void.size == 0:
        return 0
    return int(non_void[-1] + 1)


def trimmed_row(row: np.ndarray, void_code: int = 0) -> np.ndarray:
    return row[: trailing_length(row, void_code=void_code)]


def second_smallest(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    ordered = np.sort(values)
    return float(ordered[1])


def rolling_breaks(length: int, step: int, overlap: bool) -> list[tuple[int, int]]:
    if step <= 0:
        raise ValueError("'step' must be a positive integer.")
    if step == 1:
        return [(index, index + 1) for index in range(length)]
    if step >= length:
        return [(0, length)]

    starts = list(range(0, length, step))
    breaks: list[tuple[int, int]] = []
    if overlap:
        if step % 2 != 0:
            raise ValueError("'step' must be even when 'overlap=True'.")
        breaks.append((0, min(length, step // 2 + 1)))
    for start in starts:
        stop = min(length, start + step)
        breaks.append((start, stop))
        if overlap:
            overlap_start = min(length, start + step // 2)
            overlap_stop = min(length, overlap_start + step)
            if overlap_start < overlap_stop:
                breaks.append((overlap_start, overlap_stop))
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in breaks:
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized

