from __future__ import annotations

import numpy as np
import pandas as pd

from seqcourse import compat


def test_compat_namespace_exposes_tra_miner_style_functions(toy_wide) -> None:
    dataset = compat.seqdef(toy_wide)
    costs = compat.seqcost(dataset, method="CONSTANT", cval=2.0)
    distances = compat.seqdist(dataset, method="LCS", refseq=0)
    assert costs.sm.shape == (3, 3)
    assert distances.shape == (4,)
    assert np.isclose(distances[dataset.most_frequent_index()], 0.0)


def test_seqdef_treats_trailing_missing_as_void_by_default() -> None:
    wide = pd.DataFrame(
        [
            [pd.NA, "A", pd.NA],
            ["A", pd.NA, pd.NA],
            [pd.NA, pd.NA, pd.NA],
        ],
        columns=["t1", "t2", "t3"],
    )
    dataset = compat.seqdef(wide, missing_state="*")
    assert dataset.data[0, 0] == dataset.missing_code
    assert dataset.data[0, 2] == dataset.void_code
    assert dataset.data[1, 1] == dataset.void_code
    assert dataset.data[1, 2] == dataset.void_code
    assert np.all(dataset.data[2] == dataset.void_code)
