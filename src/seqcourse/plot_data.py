from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import state_distribution, state_frequencies
from .dataset import SequenceDataset
from .results import RepresentativeSequencesResult


def distribution_plot_data(seqdata: SequenceDataset | object, *, weighted: bool = True, with_missing: bool = False) -> pd.DataFrame:
    result = state_distribution(seqdata, weighted=weighted, with_missing=with_missing)
    frame = result.frequencies.T.reset_index(names="time")
    return frame.melt(id_vars="time", var_name="state", value_name="proportion")


def frequency_plot_data(seqdata: SequenceDataset | object, *, weighted: bool = True, with_missing: bool = False) -> pd.DataFrame:
    frame = state_frequencies(seqdata, weighted=weighted, with_missing=with_missing).reset_index(names="state")
    return frame


def index_plot_data(seqdata: SequenceDataset | object, *, max_sequences: int | None = None) -> dict[str, object]:
    dataset = seqdata if isinstance(seqdata, SequenceDataset) else SequenceDataset.from_wide(seqdata)
    matrix = dataset.data.copy()
    ids = list(dataset.sequence_ids)
    if max_sequences is not None and dataset.n_sequences > max_sequences:
        matrix = matrix[:max_sequences]
        ids = ids[:max_sequences]
    return {
        "matrix": matrix,
        "sequence_ids": ids,
        "time_labels": dataset.time_labels,
        "colors": (dataset.void_color,) + dataset.colors(with_missing=dataset.has_missing),
    }


def representative_plot_data(result: RepresentativeSequencesResult) -> dict[str, object]:
    dataset = result.representatives
    return {
        "matrix": dataset.data,
        "sequence_ids": list(dataset.sequence_ids),
        "time_labels": dataset.time_labels,
        "colors": (dataset.void_color,) + dataset.colors(with_missing=dataset.has_missing),
        "quality": result.quality,
    }

