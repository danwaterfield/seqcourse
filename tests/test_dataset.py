from __future__ import annotations

import pandas as pd

from seqcourse import SequenceDataset


def test_from_wide_and_to_wide_roundtrip(toy_wide: pd.DataFrame) -> None:
    dataset = SequenceDataset.from_wide(toy_wide)
    restored = dataset.to_wide(void_value=dataset.void_state)
    assert restored.shape == toy_wide.shape
    assert restored.iloc[0, 0] == "A"
    assert restored.iloc[1, 1] == "C"


def test_spell_roundtrip(toy_sequences: SequenceDataset) -> None:
    spells = toy_sequences.to_spell()
    restored = SequenceDataset.from_spell(spells)
    assert restored.data.shape == toy_sequences.data.shape
    assert restored.alphabet == toy_sequences.alphabet


def test_most_frequent_sequence_uses_weights(toy_sequences: SequenceDataset) -> None:
    assert toy_sequences.most_frequent_index() == 0
