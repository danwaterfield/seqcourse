from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seqcourse import SequenceDataset, cost_matrix, distance_matrix, representative_sequences, state_distribution


@pytest.mark.skipif(
    not Path("tests/goldens/traminer_reference.json").exists(),
    reason="Golden parity fixture has not been generated yet.",
)
def test_traminer_reference_fixture_matches_core_outputs() -> None:
    payload = json.loads(Path("tests/goldens/traminer_reference.json").read_text())
    for name, item in payload["datasets"].items():
        frame = pd.DataFrame(item["wide"], columns=item["columns"])
        dataset = SequenceDataset.from_wide(frame)
        costs = cost_matrix(dataset, method="TRATE")
        distances = distance_matrix(dataset, method="OM", sm=costs)
        stats = state_distribution(dataset)
        reps = representative_sequences(dataset, criterion="freq", diss=distance_matrix(dataset, method="LCS"))
        assert np.allclose(costs.sm, np.asarray(item["trate_costs"], dtype=float), atol=1e-6)
        assert np.allclose(distances, np.asarray(item["om_distances"], dtype=float), atol=1e-6)
        assert np.allclose(stats.frequencies.to_numpy(dtype=float), np.asarray(item["state_distribution"], dtype=float), atol=1e-6)
        assert np.allclose(stats.entropy.to_numpy(dtype=float), np.asarray(item["entropy"], dtype=float), atol=1e-6)
        assert reps.indices.tolist() == item["representatives"]
