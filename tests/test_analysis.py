from __future__ import annotations

import numpy as np

from seqcourse import mean_time_in_state, state_distribution, state_frequencies, transition_rates


def test_state_distribution_columns_sum_to_one(toy_sequences) -> None:
    result = state_distribution(toy_sequences)
    column_sums = result.frequencies.sum(axis=0).to_numpy()
    assert np.allclose(column_sums, np.ones_like(column_sums))


def test_state_frequencies_include_counts_and_proportions(toy_sequences) -> None:
    result = state_frequencies(toy_sequences)
    assert set(result.columns) == {"count", "proportion"}
    assert np.isclose(result["proportion"].sum(), 1.0)


def test_transition_rates_shape_for_static_and_time_varying(toy_sequences) -> None:
    static = transition_rates(toy_sequences)
    time_varying = transition_rates(toy_sequences, time_varying=True)
    assert static.shape == (3, 3)
    assert time_varying.shape == (3, 3, 3)


def test_mean_time_in_state_can_return_proportions(toy_sequences) -> None:
    result = mean_time_in_state(toy_sequences, prop=True)
    assert np.isclose(result.sum(), 1.0)


def test_missing_aware_summaries_include_missing_state(weighted_missing_sequences) -> None:
    distribution = state_distribution(weighted_missing_sequences, with_missing=True)
    mean_time = mean_time_in_state(weighted_missing_sequences, with_missing=True)
    transitions = transition_rates(weighted_missing_sequences, with_missing=True)
    assert weighted_missing_sequences.missing_state in distribution.frequencies.index
    assert weighted_missing_sequences.missing_state in mean_time.index
    assert transitions.shape == (3, 3)
