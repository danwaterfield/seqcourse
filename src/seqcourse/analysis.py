from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .dataset import SequenceDataset
from .results import StateDistributionResult
from ._utils import entropy


def _ensure_dataset(seqdata: SequenceDataset | object) -> SequenceDataset:
    if isinstance(seqdata, SequenceDataset):
        return seqdata
    return SequenceDataset.from_wide(seqdata)


def _state_codes(dataset: SequenceDataset, with_missing: bool) -> tuple[np.ndarray, tuple[str, ...]]:
    resolved_with_missing = with_missing and dataset.has_missing
    codes = np.arange(1, dataset.n_states + 1, dtype=np.uint16)
    states = dataset.alphabet
    if resolved_with_missing:
        codes = np.append(codes, dataset.missing_code)
        states = states + (dataset.missing_state,)
    return codes, states


def state_distribution(
    seqdata: SequenceDataset | object,
    *,
    weighted: bool = True,
    with_missing: bool = False,
    normalize_entropy: bool = True,
) -> StateDistributionResult:
    dataset = _ensure_dataset(seqdata)
    codes, states = _state_codes(dataset, with_missing=with_missing)
    weights = dataset.weights if weighted else np.ones(dataset.n_sequences, dtype=float)

    counts = np.zeros((len(states), dataset.n_positions), dtype=float)
    for state_index, code in enumerate(codes):
        counts[state_index] = np.sum((dataset.data == code) * weights[:, None], axis=0)

    valid = np.sum(
        ((dataset.data != dataset.void_code) & ((dataset.data != dataset.missing_code) | with_missing)) * weights[:, None],
        axis=0,
    )

    frequencies = np.zeros_like(counts)
    nonzero = valid > 0
    frequencies[:, nonzero] = counts[:, nonzero] / valid[nonzero]

    entropies = np.asarray([entropy(frequencies[:, column]) for column in range(dataset.n_positions)], dtype=float)
    if normalize_entropy and len(states) > 1:
        max_entropy = np.log(len(states))
        if max_entropy > 0:
            entropies = entropies / max_entropy

    frequency_frame = pd.DataFrame(frequencies, index=states, columns=dataset.time_labels)
    valid_series = pd.Series(valid, index=dataset.time_labels, name="valid_states")
    entropy_series = pd.Series(entropies, index=dataset.time_labels, name="entropy")
    return StateDistributionResult(
        frequencies=frequency_frame,
        valid_states=valid_series,
        entropy=entropy_series,
    )


def state_frequencies(
    seqdata: SequenceDataset | object,
    *,
    weighted: bool = True,
    with_missing: bool = False,
) -> pd.DataFrame:
    dataset = _ensure_dataset(seqdata)
    codes, states = _state_codes(dataset, with_missing=with_missing)
    weights = dataset.weights if weighted else np.ones(dataset.n_sequences, dtype=float)

    counts = np.zeros(len(states), dtype=float)
    for index, code in enumerate(codes):
        counts[index] = np.sum((dataset.data == code) * weights[:, None])
    proportions = counts / counts.sum() if counts.sum() else np.zeros_like(counts)
    return pd.DataFrame({"count": counts, "proportion": proportions}, index=states)


def mean_time_in_state(
    seqdata: SequenceDataset | object,
    *,
    weighted: bool = True,
    with_missing: bool = False,
    prop: bool = False,
) -> pd.Series:
    dataset = _ensure_dataset(seqdata)
    codes, states = _state_codes(dataset, with_missing=with_missing)
    weights = dataset.weights if weighted else np.ones(dataset.n_sequences, dtype=float)
    total_weight = float(weights.sum()) if weights.sum() else 1.0

    means = np.zeros(len(states), dtype=float)
    for index, code in enumerate(codes):
        per_sequence = np.sum(dataset.data == code, axis=1)
        means[index] = float(np.dot(per_sequence, weights) / total_weight)
    if prop and means.sum() > 0:
        means = means / means.sum()
    return pd.Series(means, index=states, name="mean_time")


def transition_rates(
    seqdata: SequenceDataset | object,
    sel_states: Sequence[str] | None = None,
    *,
    time_varying: bool = False,
    weighted: bool = True,
    lag: int = 1,
    with_missing: bool = False,
    count: bool = False,
) -> np.ndarray:
    dataset = _ensure_dataset(seqdata)
    available_codes, available_states = _state_codes(dataset, with_missing=with_missing)
    state_lookup = {state: int(code) for state, code in zip(available_states, available_codes, strict=True)}

    if sel_states is None:
        states = available_states
        codes = available_codes
    else:
        states = tuple(sel_states)
        codes = np.asarray([state_lookup[state] for state in states], dtype=np.uint16)

    weights = dataset.weights if weighted else np.ones(dataset.n_sequences, dtype=float)
    width = dataset.n_positions
    if lag == 0:
        raise ValueError("'lag' must be non-zero.")

    if lag > 0:
        source_positions = list(range(0, max(width - lag, 0)))
        destination_shift = lag
    else:
        source_positions = list(range(abs(lag), width))
        destination_shift = lag

    if time_varying:
        result = np.zeros((len(states), len(states), len(source_positions)), dtype=float)
        for position_index, source_position in enumerate(source_positions):
            destination_position = source_position + destination_shift
            dest_valid = dataset.data[:, destination_position] != dataset.void_code
            if not with_missing:
                dest_valid &= dataset.data[:, destination_position] != dataset.missing_code
            for row_index, source_code in enumerate(codes):
                source_match = dataset.data[:, source_position] == source_code
                denominator = float(weights[source_match & dest_valid].sum())
                if denominator == 0:
                    continue
                for column_index, destination_code in enumerate(codes):
                    numerator = float(weights[source_match & (dataset.data[:, destination_position] == destination_code)].sum())
                    result[row_index, column_index, position_index] = numerator if count else numerator / denominator
        return result

    result = np.zeros((len(states), len(states)), dtype=float)
    for row_index, source_code in enumerate(codes):
        denominator = 0.0
        numerators = np.zeros(len(states), dtype=float)
        for source_position in source_positions:
            destination_position = source_position + destination_shift
            destination = dataset.data[:, destination_position]
            valid = destination != dataset.void_code
            if not with_missing:
                valid &= destination != dataset.missing_code
            source_match = dataset.data[:, source_position] == source_code
            denominator += float(np.sum(weights[source_match & valid]))
            for column_index, destination_code in enumerate(codes):
                numerators[column_index] += float(
                    np.sum(weights[source_match & (destination == destination_code)])
                )
        if denominator == 0:
            continue
        result[row_index] = numerators if count else numerators / denominator
    return result

