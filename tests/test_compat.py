from __future__ import annotations

import numpy as np

from seqcourse import compat


def test_compat_namespace_exposes_tra_miner_style_functions(toy_wide) -> None:
    dataset = compat.seqdef(toy_wide)
    costs = compat.seqcost(dataset, method="CONSTANT", cval=2.0)
    distances = compat.seqdist(dataset, method="LCS", refseq=0)
    assert costs.sm.shape == (3, 3)
    assert distances.shape == (4,)
    assert np.isclose(distances[dataset.most_frequent_index()], 0.0)
