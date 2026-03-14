from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .dataset import SequenceDataset


@dataclass(frozen=True, slots=True)
class CostMatrixResult:
    substitution_costs: np.ndarray
    indel: float | np.ndarray
    states: tuple[str, ...]
    method: str
    time_varying: bool = False
    miss_cost: float | None = None

    @property
    def sm(self) -> np.ndarray:
        return self.substitution_costs


@dataclass(frozen=True, slots=True)
class StateDistributionResult:
    frequencies: pd.DataFrame
    valid_states: pd.Series
    entropy: pd.Series


@dataclass(frozen=True, slots=True)
class RepresentativeSequencesResult:
    dataset: SequenceDataset
    indices: np.ndarray
    scores: pd.Series
    distances: pd.DataFrame
    groups: pd.Series
    statistics: pd.DataFrame
    quality: float
    criterion: str
    radius: float
    coverage: float | None

    @property
    def representatives(self) -> SequenceDataset:
        return self.dataset.take(self.indices.tolist())
