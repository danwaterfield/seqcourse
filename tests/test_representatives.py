from __future__ import annotations

import numpy as np

from seqcourse import distance_matrix, representative_sequences


def test_frequency_based_representatives_pick_the_duplicate_sequence(toy_sequences) -> None:
    diss = distance_matrix(toy_sequences, method="LCS")
    result = representative_sequences(toy_sequences, criterion="freq", diss=diss, coverage=0.5)
    assert result.indices[0] in {0, 3}
    assert result.statistics.index[-1] == "Total"
    assert 0.0 <= result.quality <= 1.0


def test_density_based_representatives_assign_all_sequences(toy_sequences) -> None:
    diss = distance_matrix(toy_sequences, method="LCS")
    result = representative_sequences(toy_sequences, criterion="density", diss=diss, coverage=0.5)
    assert np.array_equal(np.sort(result.groups.unique()), np.arange(result.indices.shape[0]))


def test_representatives_support_missing_aware_distance_matrices(weighted_missing_sequences) -> None:
    diss = distance_matrix(weighted_missing_sequences, method="LCS", with_missing=True)
    result = representative_sequences(weighted_missing_sequences, criterion="freq", diss=diss, coverage=0.5)
    assert result.distances.shape[0] == weighted_missing_sequences.n_sequences
    assert result.indices.size >= 1
