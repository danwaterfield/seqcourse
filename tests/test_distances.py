from __future__ import annotations

import numpy as np
import pytest

from seqcourse import cost_matrix, distance_matrix


def test_lcs_distance_matrix_matches_expected_values(toy_sequences) -> None:
    result = distance_matrix(toy_sequences, method="LCS")
    assert np.array_equal(
        result,
        np.array(
            [
                [0.0, 4.0, 4.0, 0.0],
                [4.0, 0.0, 6.0, 4.0],
                [4.0, 6.0, 0.0, 4.0],
                [0.0, 4.0, 4.0, 0.0],
            ]
        ),
    )


def test_om_distance_with_constant_costs_is_symmetric(toy_sequences) -> None:
    costs = cost_matrix(toy_sequences, method="CONSTANT", cval=2.0)
    result = distance_matrix(toy_sequences, method="OM", sm=costs, indel=1.0)
    assert result.shape == (4, 4)
    assert np.allclose(result, result.T)
    assert np.allclose(np.diag(result), 0.0)


def test_ham_distance_defaults_to_unit_substitution_cost(toy_sequences) -> None:
    result = distance_matrix(toy_sequences, method="HAM")
    assert np.array_equal(
        result,
        np.array(
            [
                [0.0, 2.0, 2.0, 0.0],
                [2.0, 0.0, 3.0, 2.0],
                [2.0, 3.0, 0.0, 2.0],
                [0.0, 2.0, 2.0, 0.0],
            ]
        ),
    )


def test_distance_matrix_supports_condensed_output_and_reference_mode(toy_sequences) -> None:
    condensed = distance_matrix(toy_sequences, method="LCS", full_matrix=False)
    reference = distance_matrix(toy_sequences, method="LCS", refseq=0)
    assert condensed.shape == (6,)
    assert reference.shape == (4,)
    assert reference[0] == 0.0


def test_dhd_and_chi2_are_available(toy_sequences) -> None:
    dhd = distance_matrix(toy_sequences, method="DHD")
    chi2 = distance_matrix(toy_sequences, method="CHI2")
    euclid = distance_matrix(toy_sequences, method="EUCLID")
    assert dhd.shape == (4, 4)
    assert chi2.shape == (4, 4)
    assert euclid.shape == (4, 4)


def test_distance_matrix_accepts_boolean_norm_and_most_frequent_reference(toy_sequences) -> None:
    normalized = distance_matrix(toy_sequences, method="LCS", norm=True)
    reference = distance_matrix(toy_sequences, method="LCS", refseq="most_frequent")
    assert normalized.shape == (4, 4)
    assert reference.shape == (4,)
    assert reference[toy_sequences.most_frequent_index()] == 0.0


def test_chi2_and_euclid_reject_incompatible_norms(toy_sequences) -> None:
    with pytest.raises(ValueError):
        distance_matrix(toy_sequences, method="CHI2", norm="maxlength")
    with pytest.raises(ValueError):
        distance_matrix(toy_sequences, method="EUCLID", norm="gmean")


def test_missing_sequences_require_with_missing_for_distance(missing_sequences) -> None:
    with pytest.raises(ValueError):
        distance_matrix(missing_sequences, method="LCS")
    result = distance_matrix(missing_sequences, method="LCS", with_missing=True)
    assert result.shape == (3, 3)
