from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from .analysis import mean_time_in_state
from .backends import get_backend
from .costs import cost_matrix
from .dataset import SequenceDataset
from .results import CostMatrixResult
from ._utils import condensed_from_square, normalize_method, rolling_breaks, second_smallest, trimmed_row


def _ensure_dataset(seqdata: SequenceDataset | object) -> SequenceDataset:
    if isinstance(seqdata, SequenceDataset):
        return seqdata
    return SequenceDataset.from_wide(seqdata)


def _resolved_with_missing(dataset: SequenceDataset, with_missing: bool) -> bool:
    if with_missing and not dataset.has_missing:
        return False
    if dataset.has_missing and not with_missing:
        raise ValueError("Missing values are present; set 'with_missing=True' to include them explicitly.")
    return with_missing


def _resolve_costs(
    dataset: SequenceDataset,
    method_name: str,
    sm: CostMatrixResult | np.ndarray | str | None,
    indel: str | float | Sequence[float],
    *,
    with_missing: bool,
    weighted: bool,
) -> tuple[np.ndarray | None, np.ndarray | float | None]:
    resolved_sm = sm
    result: CostMatrixResult | None = None

    if isinstance(resolved_sm, CostMatrixResult):
        result = resolved_sm
        resolved_sm = result.sm
    elif isinstance(resolved_sm, str):
        time_varying = method_name == "DHD"
        if method_name == "DHD" and normalize_method(resolved_sm) == "CONSTANT":
            raise ValueError("'CONSTANT' is not a meaningful substitution method for DHD.")
        default_cval = 4.0 if method_name == "DHD" and normalize_method(resolved_sm) == "TRATE" else None
        if method_name == "HAM" and normalize_method(resolved_sm) == "CONSTANT":
            default_cval = 1.0
        elif method_name in {"OM", "HAM"} and normalize_method(resolved_sm) == "TRATE":
            default_cval = 2.0
        elif method_name == "OM" and normalize_method(resolved_sm) == "CONSTANT":
            default_cval = 2.0
        result = cost_matrix(
            dataset,
            method=resolved_sm,
            with_missing=with_missing,
            weighted=weighted,
            time_varying=time_varying,
            cval=default_cval,
        )
        resolved_sm = result.sm
    elif resolved_sm is None:
        if method_name == "HAM":
            result = cost_matrix(dataset, method="CONSTANT", cval=1.0, with_missing=with_missing)
            resolved_sm = result.sm
        elif method_name == "DHD":
            result = cost_matrix(dataset, method="TRATE", cval=4.0, with_missing=with_missing, weighted=weighted, time_varying=True)
            resolved_sm = result.sm
        elif method_name == "LCS":
            result = cost_matrix(dataset, method="CONSTANT", cval=2.0, with_missing=with_missing)
            resolved_sm = result.sm

    resolved_indel: np.ndarray | float | None
    if isinstance(indel, str):
        if indel != "auto":
            raise ValueError("Only 'auto' or numeric indel values are supported.")
        if result is not None:
            resolved_indel = result.indel
        elif resolved_sm is None:
            resolved_indel = None
        else:
            max_cost = float(np.max(resolved_sm))
            resolved_indel = max_cost / 2.0
    else:
        resolved_indel = np.asarray(indel, dtype=float) if isinstance(indel, Sequence) else float(indel)

    return resolved_sm, resolved_indel


def _scalar_or_vector_indel(indel: np.ndarray | float | None, code: int) -> float:
    if indel is None:
        raise ValueError("An indel cost is required for this distance.")
    if np.isscalar(indel):
        return float(indel)
    return float(np.asarray(indel, dtype=float)[code - 1])


def _substitution_cost(sm: np.ndarray, left: int, right: int, position: int | None = None) -> float:
    if left == right:
        return 0.0
    if sm.ndim == 2:
        return float(sm[left - 1, right - 1])
    if position is None:
        raise ValueError("A position is required for time-varying substitution costs.")
    position_index = min(position, sm.shape[2] - 1)
    return float(sm[left - 1, right - 1, position_index])


def _om_distance(seq1: np.ndarray, seq2: np.ndarray, sm: np.ndarray, indel: np.ndarray | float) -> tuple[float, float]:
    len1 = len(seq1)
    len2 = len(seq2)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=float)
    for index in range(1, len1 + 1):
        dp[index, 0] = dp[index - 1, 0] + _scalar_or_vector_indel(indel, int(seq1[index - 1]))
    for index in range(1, len2 + 1):
        dp[0, index] = dp[0, index - 1] + _scalar_or_vector_indel(indel, int(seq2[index - 1]))
    for left in range(1, len1 + 1):
        for right in range(1, len2 + 1):
            delete = dp[left - 1, right] + _scalar_or_vector_indel(indel, int(seq1[left - 1]))
            insert = dp[left, right - 1] + _scalar_or_vector_indel(indel, int(seq2[right - 1]))
            substitute = dp[left - 1, right - 1] + _substitution_cost(sm, int(seq1[left - 1]), int(seq2[right - 1]))
            dp[left, right] = min(delete, insert, substitute)
    max_sub = float(np.max(sm)) if sm.size else 0.0
    max_indel = float(np.max(indel)) if not np.isscalar(indel) else float(indel)
    max_distance = max(max_sub, 2.0 * max_indel) * max(len1, len2, 1)
    return float(dp[len1, len2]), max_distance


def _ham_distance(seq1: np.ndarray, seq2: np.ndarray, sm: np.ndarray) -> tuple[float, float]:
    max_length = max(len(seq1), len(seq2))
    if max_length == 0:
        return 0.0, 1.0
    max_cost = float(np.max(sm))
    distance = 0.0
    for position in range(max_length):
        if position >= len(seq1) or position >= len(seq2):
            distance += max_cost
            continue
        distance += _substitution_cost(sm, int(seq1[position]), int(seq2[position]))
    return distance, max_cost * max_length


def _dhd_distance(seq1: np.ndarray, seq2: np.ndarray, sm: np.ndarray) -> tuple[float, float]:
    max_length = max(len(seq1), len(seq2))
    if max_length == 0:
        return 0.0, 1.0
    distance = 0.0
    max_cost = 0.0
    for position in range(max_length):
        slice_max = float(np.max(sm[:, :, min(position, sm.shape[2] - 1)]))
        max_cost += slice_max
        if position >= len(seq1) or position >= len(seq2):
            distance += slice_max
            continue
        distance += _substitution_cost(sm, int(seq1[position]), int(seq2[position]), position=position)
    return distance, max_cost


def _lcs_length(seq1: np.ndarray, seq2: np.ndarray) -> int:
    previous = np.zeros(len(seq2) + 1, dtype=int)
    for left in seq1:
        current = np.zeros(len(seq2) + 1, dtype=int)
        for right_index, right in enumerate(seq2, start=1):
            if left == right:
                current[right_index] = previous[right_index - 1] + 1
            else:
                current[right_index] = max(previous[right_index], current[right_index - 1])
        previous = current
    return int(previous[-1])


def _prefix_length(seq1: np.ndarray, seq2: np.ndarray) -> int:
    match = 0
    for left, right in zip(seq1, seq2, strict=False):
        if left != right:
            break
        match += 1
    return match


def _suffix_length(seq1: np.ndarray, seq2: np.ndarray) -> int:
    match = 0
    for left, right in zip(seq1[::-1], seq2[::-1], strict=False):
        if left != right:
            break
        match += 1
    return match


def _sequence_distance(method_name: str, seq1: np.ndarray, seq2: np.ndarray, sm: np.ndarray | None, indel: np.ndarray | float | None) -> tuple[float, float]:
    if method_name == "OM":
        if sm is None or indel is None:
            raise ValueError("OM distance requires substitution costs and indel costs.")
        return _om_distance(seq1, seq2, sm, indel)
    if method_name == "HAM":
        if sm is None:
            raise ValueError("HAM distance requires substitution costs.")
        return _ham_distance(seq1, seq2, sm)
    if method_name == "DHD":
        if sm is None or sm.ndim != 3:
            raise ValueError("DHD distance requires time-varying substitution costs.")
        return _dhd_distance(seq1, seq2, sm)
    if method_name == "LCS":
        common = _lcs_length(seq1, seq2)
    elif method_name == "LCP":
        common = _prefix_length(seq1, seq2)
    elif method_name == "RLCP":
        common = _suffix_length(seq1, seq2)
    else:
        raise ValueError(f"Unsupported sequence distance method {method_name!r}.")
    raw = float(len(seq1) + len(seq2) - 2 * common)
    return raw, float(max(len(seq1) + len(seq2), 1))


def _normalized_distance(
    raw: float,
    *,
    norm: str,
    method_name: str,
    len1: int,
    len2: int,
    max_distance: float,
) -> float:
    norm_name = normalize_method(norm)
    if norm_name == "AUTO":
        if method_name in {"OM", "HAM", "DHD"}:
            norm_name = "MAXLENGTH"
        elif method_name in {"LCS", "LCP", "RLCP"}:
            norm_name = "GMEAN"
        else:
            norm_name = "NONE"
    if norm_name == "NONE":
        return raw
    if raw == 0:
        return 0.0
    if norm_name == "MAXLENGTH":
        return raw / max(len1, len2, 1)
    if norm_name == "GMEAN":
        denominator = math.sqrt(max(len1, 1) * max(len2, 1))
        return raw / denominator
    if norm_name == "MAXDIST":
        return raw / max(max_distance, 1.0)
    if norm_name == "YUJIANBO":
        return (2.0 * raw) / (len1 + len2 + raw)
    raise ValueError(f"Unsupported normalization {norm!r}.")


def _prepare_sequences(dataset: SequenceDataset) -> list[np.ndarray]:
    return [trimmed_row(row, void_code=dataset.void_code) for row in dataset.data]


def _build_chi2_windows(length: int, breaks: Sequence[tuple[int, int]] | None, step: int, overlap: bool) -> list[tuple[int, int]]:
    if breaks is None:
        return rolling_breaks(length, step, overlap)
    windows: list[tuple[int, int]] = []
    for start, stop in breaks:
        if start < 1 or stop < start:
            raise ValueError("Breaks must use one-based inclusive coordinates.")
        windows.append((start - 1, stop))
    return windows


def _chi2_feature_matrix(
    dataset: SequenceDataset,
    *,
    with_missing: bool,
    weighted: bool,
    breaks: Sequence[tuple[int, int]] | None,
    step: int,
    overlap: bool,
) -> tuple[np.ndarray, np.ndarray]:
    codes = list(range(1, dataset.n_states + 1))
    if with_missing and dataset.has_missing:
        codes.append(dataset.missing_code)

    windows = _build_chi2_windows(dataset.n_positions, breaks, step, overlap)
    weights = dataset.weights if weighted else np.ones(dataset.n_sequences, dtype=float)
    features: list[np.ndarray] = []
    marginals: list[np.ndarray] = []
    for start, stop in windows:
        window = dataset.data[:, start:stop]
        window_counts = np.zeros((dataset.n_sequences, len(codes)), dtype=float)
        for code_index, code in enumerate(codes):
            window_counts[:, code_index] = np.sum(window == code, axis=1)
        row_sums = window_counts.sum(axis=1, keepdims=True)
        nonzero = row_sums[:, 0] > 0
        window_counts[nonzero] = window_counts[nonzero] / row_sums[nonzero]
        features.append(window_counts)

        marginal = np.sum(window_counts * weights[:, None], axis=0)
        if marginal.sum() > 0:
            marginal = marginal / marginal.sum()
        marginals.append(marginal)
    return np.hstack(features), np.hstack(marginals)


def _chi2_or_euclid(
    dataset: SequenceDataset,
    *,
    method_name: str,
    refseq: object,
    with_missing: bool,
    weighted: bool,
    norm: str,
    breaks: Sequence[tuple[int, int]] | None,
    step: int,
    overlap: bool,
    global_pdotj: Sequence[float] | str | None,
) -> np.ndarray:
    features, marginals = _chi2_feature_matrix(
        dataset,
        with_missing=with_missing,
        weighted=weighted and method_name != "EUCLID",
        breaks=breaks,
        step=step,
        overlap=overlap,
    )
    if global_pdotj == "obs":
        observed = mean_time_in_state(dataset, weighted=weighted, with_missing=with_missing, prop=True).to_numpy(dtype=float)
        repeats = features.shape[1] // len(observed) if len(observed) else 0
        weights_vector = np.tile(observed, repeats)
    elif global_pdotj is not None:
        vector = np.asarray(global_pdotj, dtype=float)
        vector = vector / vector.sum()
        repeats = features.shape[1] // len(vector) if len(vector) else 0
        weights_vector = np.tile(vector, repeats)
    else:
        weights_vector = marginals
    weights_vector = np.asarray(weights_vector, dtype=float)
    positive = weights_vector > 0
    features = features[:, positive]
    weights_vector = weights_vector[positive]

    def pair(left: int, right: int) -> float:
        diff = features[left] - features[right]
        if method_name == "EUCLID":
            value = float(np.sqrt(np.sum(diff ** 2)))
        else:
            value = float(np.sqrt(np.sum((diff ** 2) / weights_vector)))
        if normalize_method(norm) != "NONE":
            value = value / math.sqrt(max(features.shape[1], 1))
        return value

    if refseq is None:
        matrix = np.zeros((dataset.n_sequences, dataset.n_sequences), dtype=float)
        for left in range(dataset.n_sequences):
            for right in range(left + 1, dataset.n_sequences):
                value = pair(left, right)
                matrix[left, right] = value
                matrix[right, left] = value
        return matrix

    if isinstance(refseq, tuple) and len(refseq) == 2:
        left_indices = np.asarray(refseq[0], dtype=int)
        right_indices = np.asarray(refseq[1], dtype=int)
        matrix = np.zeros((len(left_indices), len(right_indices)), dtype=float)
        for left_position, left_index in enumerate(left_indices):
            for right_position, right_index in enumerate(right_indices):
                matrix[left_position, right_position] = pair(int(left_index), int(right_index))
        return matrix

    reference_index = int(refseq)
    return np.asarray([pair(index, reference_index) for index in range(dataset.n_sequences)], dtype=float)


def _resolve_reference(dataset: SequenceDataset, refseq: object | None) -> tuple[str, object | None]:
    if refseq is None:
        return "pairwise", None
    if isinstance(refseq, SequenceDataset):
        if refseq.n_sequences != 1:
            raise ValueError("Reference SequenceDataset objects must contain exactly one sequence.")
        extended = np.vstack([dataset.data, refseq.data])
        augmented = SequenceDataset(
            data=extended,
            alphabet=dataset.alphabet,
            state_labels=dataset.state_labels,
            state_colors=dataset.state_colors,
            weights=np.concatenate([dataset.weights, np.zeros(1, dtype=float)]),
            time_labels=dataset.time_labels,
            sequence_ids=dataset.sequence_ids + ("ref",),
            missing_state=dataset.missing_state,
            missing_label=dataset.missing_label,
            missing_color=dataset.missing_color,
            void_state=dataset.void_state,
            void_label=dataset.void_label,
            void_color=dataset.void_color,
        )
        return "external", augmented
    if isinstance(refseq, tuple) and len(refseq) == 2:
        return "sets", (np.asarray(refseq[0], dtype=int), np.asarray(refseq[1], dtype=int))
    if isinstance(refseq, list) and len(refseq) == 2:
        return "sets", (np.asarray(refseq[0], dtype=int), np.asarray(refseq[1], dtype=int))
    if isinstance(refseq, int):
        if refseq < 0 or refseq >= dataset.n_sequences:
            raise IndexError("Reference sequence index out of bounds.")
        return "index", int(refseq)
    raise TypeError("Unsupported refseq value.")


def _distance_matrix_impl(
    seqdata: SequenceDataset | object,
    *,
    method: str,
    refseq: object | None = None,
    norm: str = "none",
    indel: str | float | Sequence[float] = "auto",
    sm: CostMatrixResult | np.ndarray | str | None = None,
    with_missing: bool = False,
    full_matrix: bool = True,
    weighted: bool = True,
    breaks: Sequence[tuple[int, int]] | None = None,
    step: int = 1,
    overlap: bool = False,
    global_pdotj: Sequence[float] | str | None = None,
) -> np.ndarray:
    dataset = _ensure_dataset(seqdata)
    with_missing = _resolved_with_missing(dataset, with_missing)
    method_name = normalize_method(method)
    supported = {"OM", "HAM", "DHD", "LCS", "LCP", "RLCP", "CHI2", "EUCLID"}
    if method_name not in supported:
        raise ValueError(f"Unsupported distance method {method!r}. Supported methods: {', '.join(sorted(supported))}.")

    mode, reference = _resolve_reference(dataset, refseq)
    if mode == "external":
        dataset = reference
        reference = dataset.n_sequences - 1
        mode = "index"

    if method_name in {"CHI2", "EUCLID"}:
        result = _chi2_or_euclid(
            dataset,
            method_name=method_name,
            refseq=reference,
            with_missing=with_missing,
            weighted=weighted,
            norm=norm,
            breaks=breaks,
            step=step,
            overlap=overlap,
            global_pdotj=global_pdotj,
        )
        if mode == "pairwise" and not full_matrix:
            return condensed_from_square(result)
        if mode == "index" and dataset.sequence_ids[-1] == "ref":
            return result[:-1]
        return result

    substitution_costs, resolved_indel = _resolve_costs(
        dataset,
        method_name,
        sm,
        indel,
        with_missing=with_missing,
        weighted=weighted,
    )

    sequences = _prepare_sequences(dataset)

    def pair(left_index: int, right_index: int) -> float:
        left = sequences[left_index]
        right = sequences[right_index]
        raw, max_distance = _sequence_distance(method_name, left, right, substitution_costs, resolved_indel)
        return _normalized_distance(
            raw,
            norm=norm,
            method_name=method_name,
            len1=len(left),
            len2=len(right),
            max_distance=max_distance,
        )

    if mode == "pairwise":
        matrix = np.zeros((dataset.n_sequences, dataset.n_sequences), dtype=float)
        for left in range(dataset.n_sequences):
            for right in range(left + 1, dataset.n_sequences):
                value = pair(left, right)
                matrix[left, right] = value
                matrix[right, left] = value
        return matrix if full_matrix else condensed_from_square(matrix)

    if mode == "sets":
        left_indices, right_indices = reference
        matrix = np.zeros((len(left_indices), len(right_indices)), dtype=float)
        for left_position, left_index in enumerate(left_indices):
            for right_position, right_index in enumerate(right_indices):
                matrix[left_position, right_position] = pair(int(left_index), int(right_index))
        return matrix

    reference_index = int(reference)
    result = np.asarray([pair(index, reference_index) for index in range(dataset.n_sequences)], dtype=float)
    if dataset.sequence_ids[-1] == "ref":
        return result[:-1]
    return result


def distance_matrix(
    seqdata: SequenceDataset | object,
    *,
    method: str,
    refseq: object | None = None,
    norm: str = "none",
    indel: str | float | Sequence[float] = "auto",
    sm: CostMatrixResult | np.ndarray | str | None = None,
    with_missing: bool = False,
    full_matrix: bool = True,
    weighted: bool = True,
    breaks: Sequence[tuple[int, int]] | None = None,
    step: int = 1,
    overlap: bool = False,
    global_pdotj: Sequence[float] | str | None = None,
    backend: str | None = None,
) -> np.ndarray:
    engine = get_backend(backend)
    return engine.compute_distance_matrix(
        seqdata,
        method=method,
        refseq=refseq,
        norm=norm,
        indel=indel,
        sm=sm,
        with_missing=with_missing,
        full_matrix=full_matrix,
        weighted=weighted,
        breaks=breaks,
        step=step,
        overlap=overlap,
        global_pdotj=global_pdotj,
    )


def seqdist(seqdata: SequenceDataset | object, *args: object, **kwargs: object) -> np.ndarray:
    return distance_matrix(seqdata, *args, **kwargs)

