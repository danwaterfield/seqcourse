from __future__ import annotations

import numpy as np

from seqcourse import cost_matrix


def test_constant_cost_matrix_is_square_and_zero_diagonal(toy_sequences) -> None:
    result = cost_matrix(toy_sequences, method="CONSTANT", cval=2.0)
    assert result.sm.shape == (3, 3)
    assert np.allclose(np.diag(result.sm), 0.0)
    assert np.allclose(result.sm[0, 1], 2.0)


def test_trate_cost_matrix_is_symmetric(toy_sequences) -> None:
    result = cost_matrix(toy_sequences, method="TRATE")
    assert np.allclose(result.sm, result.sm.T)


def test_indelslog_returns_vector_indel_costs(toy_sequences) -> None:
    result = cost_matrix(toy_sequences, method="INDELSLOG")
    assert result.sm.shape == (3, 3)
    assert np.asarray(result.indel).shape == (3,)


def test_constant_costs_apply_explicit_missing_cost(missing_sequences) -> None:
    result = cost_matrix(missing_sequences, method="CONSTANT", cval=2.0, with_missing=True, miss_cost=5.0)
    assert result.sm.shape == (3, 3)
    assert np.allclose(result.sm[-1, :-1], 5.0)
    assert np.allclose(result.sm[:-1, -1], 5.0)
    assert result.sm[-1, -1] == 0.0


def test_indelslog_can_fix_missing_substitution_costs(missing_sequences) -> None:
    fixed = cost_matrix(
        missing_sequences,
        method="INDELSLOG",
        with_missing=True,
        miss_cost=4.0,
        miss_cost_fixed=True,
    )
    assert np.allclose(fixed.sm[-1, :-1], 4.0)
    assert np.allclose(fixed.sm[:-1, -1], 4.0)
