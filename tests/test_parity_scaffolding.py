from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seqcourse import (
    SequenceDataset,
    cost_matrix,
    distance_matrix,
    mean_time_in_state,
    representative_sequences,
    state_distribution,
    transition_rates,
)


def _golden_path() -> Path:
    return Path(os.environ.get("SEQCOURSE_TRAMINER_GOLDEN", "tests/goldens/traminer_reference.json"))


def _optional_vector(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, dict) and not value:
        return None
    return value


@pytest.mark.skipif(
    not _golden_path().exists(),
    reason="Golden parity fixture has not been generated yet.",
)
def test_traminer_reference_fixture_matches_core_outputs() -> None:
    payload = json.loads(_golden_path().read_text())
    assert payload["schema_version"] >= 4
    assert payload["upstream"]["package"] == "TraMineR"
    for item in payload["datasets"].values():
        frame = pd.DataFrame(item["wide"], columns=item["columns"])
        with_missing = bool(item.get("with_missing", False))
        dataset = SequenceDataset.from_wide(frame, weights=_optional_vector(item.get("weights")))
        costs = cost_matrix(dataset, method="TRATE", with_missing=with_missing)
        distances = distance_matrix(dataset, method="OM", sm=costs, with_missing=with_missing)
        lcs_auto = distance_matrix(dataset, method="LCS", norm="auto", with_missing=with_missing)
        stats = state_distribution(dataset, with_missing=with_missing)
        reps = representative_sequences(
            dataset,
            criterion="freq",
            diss=distance_matrix(dataset, method="LCS", with_missing=with_missing),
        )
        assert np.allclose(costs.sm, np.asarray(item["trate_costs"], dtype=float), atol=1e-6)
        assert np.allclose(np.asarray(costs.indel, dtype=float), np.asarray(item["trate_indel"], dtype=float), atol=1e-6)
        assert np.allclose(distances, np.asarray(item["om_distances"], dtype=float), atol=1e-6)
        if item["ham_distances"] is not None:
            ham = distance_matrix(dataset, method="HAM", with_missing=with_missing)
            assert np.allclose(ham, np.asarray(item["ham_distances"], dtype=float), atol=1e-6)
        assert np.allclose(lcs_auto, np.asarray(item["lcs_auto"], dtype=float), atol=1e-6)
        assert np.allclose(stats.frequencies.to_numpy(dtype=float), np.asarray(item["state_distribution"], dtype=float), atol=1e-6)
        assert np.allclose(stats.entropy.to_numpy(dtype=float), np.asarray(item["entropy"], dtype=float), atol=1e-6)
        assert np.allclose(
            transition_rates(dataset, with_missing=with_missing),
            np.asarray(item["transition_rates"], dtype=float),
            atol=1e-6,
        )
        assert np.allclose(
            mean_time_in_state(dataset, with_missing=with_missing).to_numpy(dtype=float),
            np.asarray(item["mean_time"], dtype=float),
            atol=1e-6,
        )
        assert reps.indices.tolist() == item["representatives"]
        assert reps.groups.tolist() == item["representative_groups"]


def test_optional_vector_normalizes_empty_json_objects() -> None:
    assert _optional_vector(None) is None
    assert _optional_vector({}) is None
    assert _optional_vector([]) == []
    assert _optional_vector([1.0, 2.0]) == [1.0, 2.0]
