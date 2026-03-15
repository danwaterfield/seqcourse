from __future__ import annotations

import pandas as pd
import pytest

from seqcourse import SequenceDataset


@pytest.fixture()
def toy_wide() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["A", "A", "B", "B"],
            ["A", "C", "C", "B"],
            ["B", "B", "B", "B"],
            ["A", "A", "B", "B"],
        ],
        index=["s1", "s2", "s3", "s4"],
        columns=["t1", "t2", "t3", "t4"],
    )


@pytest.fixture()
def toy_sequences(toy_wide: pd.DataFrame) -> SequenceDataset:
    return SequenceDataset.from_wide(toy_wide, weights=[1.0, 1.5, 1.0, 2.0])


@pytest.fixture()
def missing_wide() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["A", pd.NA, "B"],
            ["A", "B", "B"],
            ["B", "B", pd.NA],
        ],
        index=["m1", "m2", "m3"],
        columns=["t1", "t2", "t3"],
    )


@pytest.fixture()
def missing_sequences(missing_wide: pd.DataFrame) -> SequenceDataset:
    return SequenceDataset.from_wide(missing_wide, weights=[1.0, 2.0, 1.0])


@pytest.fixture()
def weighted_missing_wide() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["A", pd.NA, "B", "B"],
            [pd.NA, pd.NA, pd.NA, pd.NA],
            ["B", "B", pd.NA, "A"],
        ],
        index=["w1", "w2", "w3"],
        columns=["t1", "t2", "t3", "t4"],
    )


@pytest.fixture()
def weighted_missing_sequences(weighted_missing_wide: pd.DataFrame) -> SequenceDataset:
    return SequenceDataset.from_wide(weighted_missing_wide, weights=[1.0, 0.0, 2.0])
