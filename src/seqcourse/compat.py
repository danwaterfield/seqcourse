from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .analysis import mean_time_in_state, state_distribution, state_frequencies, transition_rates
from .costs import cost_matrix
from .dataset import SequenceDataset
from .distances import distance_matrix
from .plotting import plot_distribution, plot_frequency, plot_index, plot_representatives
from .representatives import representative_sequences


def seqdef(
    data: object,
    cols: list[int] | list[str] | None = None,
    *,
    alphabet: list[str] | tuple[str, ...] | None = None,
    labels: list[str] | tuple[str, ...] | None = None,
    cpal: list[str] | tuple[str, ...] | None = None,
    weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    missing_values: set[object] | None = None,
    void_values: set[object] | None = None,
) -> SequenceDataset:
    if isinstance(data, SequenceDataset):
        return data
    selected = data
    if cols is not None and isinstance(data, pd.DataFrame):
        selected = data.loc[:, cols]
    elif cols is not None:
        selected = np.asarray(data)[:, cols]
    return SequenceDataset.from_wide(
        selected,
        alphabet=alphabet,
        labels=labels,
        colors=cpal,
        weights=weights,
        missing_values=missing_values,
        void_values=void_values,
    )


def seqformat(
    data: object,
    cols: list[int] | list[str] | None = None,
    *,
    from_format: str = "STS",
    to: str = "SPELL",
    id_col: str = "id",
    state_col: str = "state",
    start_col: str = "start",
    end_col: str | None = "end",
    duration_col: str | None = None,
    weight_col: str | None = None,
    **kwargs: object,
) -> SequenceDataset | pd.DataFrame:
    source = from_format.upper()
    target = to.upper()
    if source == "STS" and target == "SPELL":
        dataset = seqdef(data, cols, **kwargs)
        return dataset.to_spell(id_name=id_col, state_name=state_col, start_name=start_col, end_name=end_col or "end")
    if source == "SPELL" and target == "STS":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("SPELL input must be a pandas DataFrame.")
        return SequenceDataset.from_spell(
            data,
            id_col=id_col,
            state_col=state_col,
            start_col=start_col,
            end_col=end_col,
            duration_col=duration_col,
            weight_col=weight_col,
            **kwargs,
        )
    raise ValueError(f"Unsupported conversion from {from_format!r} to {to!r}.")


def seqcost(seqdata: SequenceDataset | object, *args: object, **kwargs: object):
    return cost_matrix(seqdata, *args, **kwargs)


def seqsubm(seqdata: SequenceDataset | object, *args: object, **kwargs: object):
    return cost_matrix(seqdata, *args, **kwargs).sm


def _compat_refseq(dataset: SequenceDataset, refseq: object | None) -> object | None:
    if refseq is None:
        return None
    if isinstance(refseq, int):
        if refseq == 0:
            return dataset.most_frequent_index()
        return refseq - 1
    if isinstance(refseq, list) and len(refseq) == 2:
        return ([value - 1 for value in refseq[0]], [value - 1 for value in refseq[1]])
    if isinstance(refseq, tuple) and len(refseq) == 2:
        return tuple([value - 1 for value in group] for group in refseq)
    return refseq


def seqdist(seqdata: SequenceDataset | object, *args: object, refseq: object | None = None, **kwargs: object):
    dataset = seqdef(seqdata) if not isinstance(seqdata, SequenceDataset) else seqdata
    translated_refseq = _compat_refseq(dataset, refseq)
    return distance_matrix(dataset, *args, refseq=translated_refseq, **kwargs)


def seqstatd(seqdata: SequenceDataset | object, *args: object, norm: bool = True, **kwargs: object):
    return state_distribution(seqdata, *args, normalize_entropy=norm, **kwargs)


def seqstatf(seqdata: SequenceDataset | object, *args: object, **kwargs: object):
    return state_frequencies(seqdata, *args, **kwargs)


def seqtrate(seqdata: SequenceDataset | object, *args: object, **kwargs: object):
    return transition_rates(seqdata, *args, **kwargs)


def seqmeant(seqdata: SequenceDataset | object, *args: object, **kwargs: object):
    return mean_time_in_state(seqdata, *args, **kwargs)


def seqrep(seqdata: SequenceDataset | object, *args: object, **kwargs: object):
    return representative_sequences(seqdata, *args, **kwargs)


def seqdplot(seqdata: SequenceDataset | object, *args: object, **kwargs: object):
    return plot_distribution(seqdata, *args, **kwargs)


def seqfplot(seqdata: SequenceDataset | object, *args: object, **kwargs: object):
    return plot_frequency(seqdata, *args, **kwargs)


def seqIplot(seqdata: SequenceDataset | object, *args: object, **kwargs: object):
    return plot_index(seqdata, *args, **kwargs)


def seqrfplot(seqdata: SequenceDataset | object, *args: object, **kwargs: object):
    return plot_representatives(seqdata, *args, **kwargs)


@dataclass(frozen=True, slots=True)
class CompatNamespace:
    seqcost: Any = staticmethod(seqcost)
    seqdef: Any = staticmethod(seqdef)
    seqdist: Any = staticmethod(seqdist)
    seqdplot: Any = staticmethod(seqdplot)
    seqformat: Any = staticmethod(seqformat)
    seqfplot: Any = staticmethod(seqfplot)
    seqIplot: Any = staticmethod(seqIplot)
    seqmeant: Any = staticmethod(seqmeant)
    seqrep: Any = staticmethod(seqrep)
    seqrfplot: Any = staticmethod(seqrfplot)
    seqstatd: Any = staticmethod(seqstatd)
    seqstatf: Any = staticmethod(seqstatf)
    seqsubm: Any = staticmethod(seqsubm)
    seqtrate: Any = staticmethod(seqtrate)


compat = CompatNamespace()
