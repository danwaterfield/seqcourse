from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from .analysis import state_distribution
from .dataset import SequenceDataset
from .plot_data import distribution_plot_data, frequency_plot_data, index_plot_data, representative_plot_data
from .representatives import representative_sequences
from .results import RepresentativeSequencesResult


def _dataset(obj: SequenceDataset | object) -> SequenceDataset:
    if isinstance(obj, SequenceDataset):
        return obj
    return SequenceDataset.from_wide(obj)


def plot_distribution(
    seqdata: SequenceDataset | object,
    *,
    ax: plt.Axes | None = None,
    weighted: bool = True,
    with_missing: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    dataset = _dataset(seqdata)
    result = state_distribution(dataset, weighted=weighted, with_missing=with_missing)
    if ax is None:
        figure, ax = plt.subplots(figsize=(10, 4))
    else:
        figure = ax.figure
    x = np.arange(dataset.n_positions)
    y = result.frequencies.to_numpy(dtype=float)
    colors = dataset.colors(with_missing=with_missing)
    ax.stackplot(x, y, labels=result.frequencies.index.tolist(), colors=colors)
    ax.set_xlim(0, dataset.n_positions - 1 if dataset.n_positions else 0)
    ax.set_ylim(0, 1)
    ax.set_xticks(x, dataset.time_labels, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("State distribution over time")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    figure.tight_layout()
    return figure, ax


def plot_frequency(
    seqdata: SequenceDataset | object,
    *,
    ax: plt.Axes | None = None,
    weighted: bool = True,
    with_missing: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    dataset = _dataset(seqdata)
    data = frequency_plot_data(dataset, weighted=weighted, with_missing=with_missing)
    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 4))
    else:
        figure = ax.figure
    colors = dataset.colors(with_missing=with_missing)
    ax.barh(data["state"], data["proportion"], color=colors)
    ax.set_xlabel("Proportion")
    ax.set_title("State frequencies")
    figure.tight_layout()
    return figure, ax


def _imshow_sequences(ax: plt.Axes, dataset: SequenceDataset, matrix: np.ndarray, sequence_ids: list[str]) -> None:
    if dataset.has_missing:
        colors = (dataset.void_color,) + dataset.state_colors + (dataset.missing_color,)
    else:
        colors = (dataset.void_color,) + dataset.state_colors
    cmap = ListedColormap(colors)
    boundaries = np.arange(len(colors) + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(sequence_ids)), sequence_ids)
    ax.set_xticks(np.arange(dataset.n_positions), dataset.time_labels, rotation=45, ha="right")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sequence")


def plot_index(
    seqdata: SequenceDataset | object,
    *,
    ax: plt.Axes | None = None,
    max_sequences: int | None = 200,
) -> tuple[plt.Figure, plt.Axes]:
    dataset = _dataset(seqdata)
    payload = index_plot_data(dataset, max_sequences=max_sequences)
    if ax is None:
        figure, ax = plt.subplots(figsize=(10, max(4, len(payload["sequence_ids"]) * 0.3)))
    else:
        figure = ax.figure
    _imshow_sequences(ax, dataset, payload["matrix"], payload["sequence_ids"])
    ax.set_title("Sequence index plot")
    figure.tight_layout()
    return figure, ax


def plot_representatives(
    seqdata: SequenceDataset | RepresentativeSequencesResult | object,
    *,
    ax: plt.Axes | None = None,
    max_sequences: int | None = None,
    **representative_kwargs: object,
) -> tuple[plt.Figure, plt.Axes]:
    if isinstance(seqdata, RepresentativeSequencesResult):
        result = seqdata
    else:
        result = representative_sequences(seqdata, **representative_kwargs)
    payload = representative_plot_data(result)
    dataset = result.representatives
    if max_sequences is not None and dataset.n_sequences > max_sequences:
        matrix = payload["matrix"][:max_sequences]
        sequence_ids = payload["sequence_ids"][:max_sequences]
    else:
        matrix = payload["matrix"]
        sequence_ids = payload["sequence_ids"]
    if ax is None:
        figure, ax = plt.subplots(figsize=(10, max(3, len(sequence_ids) * 0.4)))
    else:
        figure = ax.figure
    _imshow_sequences(ax, dataset, matrix, sequence_ids)
    ax.set_title(f"Representative sequences (quality={result.quality:.3f})")
    figure.tight_layout()
    return figure, ax

