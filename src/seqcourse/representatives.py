from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import SequenceDataset
from .distances import distance_matrix
from .results import RepresentativeSequencesResult


def _ensure_dataset(seqdata: SequenceDataset | object) -> SequenceDataset:
    if isinstance(seqdata, SequenceDataset):
        return seqdata
    return SequenceDataset.from_wide(seqdata)


def _distance_to_center(diss: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted_sums = diss @ weights
    return weighted_sums - np.average(weighted_sums, weights=weights) / 2.0


def representative_sequences(
    seqdata: SequenceDataset | object,
    *,
    criterion: str = "density",
    score: np.ndarray | None = None,
    decreasing: bool = True,
    coverage: float = 0.25,
    nrep: int | None = None,
    pradius: float = 0.10,
    dmax: float | None = None,
    diss: np.ndarray | None = None,
    weighted: bool = True,
    **distance_kwargs: object,
) -> RepresentativeSequencesResult:
    dataset = _ensure_dataset(seqdata)
    weights = dataset.weights if weighted else np.ones(dataset.n_sequences, dtype=float)
    distance_values = distance_matrix(dataset, full_matrix=True, **distance_kwargs) if diss is None else np.asarray(diss, dtype=float)
    if distance_values.shape != (dataset.n_sequences, dataset.n_sequences):
        raise ValueError("Representative sequence extraction requires a square distance matrix.")

    dmax = float(distance_values.max(initial=0.0)) if dmax is None else float(dmax)
    radius = dmax * float(pradius)

    criterion_name = criterion.lower()
    if score is None:
        if criterion_name == "density":
            neighbourhoods = distance_values < radius
            score_values = neighbourhoods @ weights
            decreasing = True
        elif criterion_name == "freq":
            score_values = (distance_values == 0.0) @ weights
            decreasing = True
        elif criterion_name == "dist":
            score_values = distance_values @ weights
            decreasing = False
        elif criterion_name == "random":
            rng = np.random.default_rng(0)
            score_values = rng.permutation(dataset.n_sequences)
            decreasing = False
        else:
            raise ValueError(f"Unknown representative criterion {criterion!r}.")
    else:
        score_values = np.asarray(score, dtype=float)
        if score_values.shape != (dataset.n_sequences,):
            raise ValueError("'score' must have one value per sequence.")

    order = np.lexsort((np.arange(dataset.n_sequences), -score_values if decreasing else score_values))
    ordered_diss = distance_values[np.ix_(order, order)]
    chosen: list[int] = []

    if nrep is None:
        represented = 0.0
        while represented < coverage and len(chosen) < dataset.n_sequences:
            candidate = len(chosen)
            while candidate < dataset.n_sequences:
                if not chosen or np.all(ordered_diss[candidate, chosen] > radius):
                    chosen.append(candidate)
                    represented_mask = np.any(ordered_diss[:, chosen] < radius, axis=1)
                    represented = float(weights[order][represented_mask].sum() / weights.sum())
                    break
                candidate += 1
            else:
                break
        resolved_coverage = represented
    else:
        candidate = 0
        while len(chosen) < nrep and candidate < dataset.n_sequences:
            if not chosen or np.all(ordered_diss[candidate, chosen] > radius):
                chosen.append(candidate)
            candidate += 1
        resolved_coverage = None

    representative_indices = order[np.asarray(chosen, dtype=int)]
    distances_to_representatives = distance_values[:, representative_indices]
    groups = np.argmin(distances_to_representatives, axis=1)
    min_distance = distances_to_representatives[np.arange(dataset.n_sequences), groups]

    total_center = _distance_to_center(distance_values, weights)
    stats_rows: list[dict[str, float]] = []
    for rep_position, representative_index in enumerate(representative_indices):
        assigned = groups == rep_position
        assigned_weights = weights[assigned]
        assigned_distances = distances_to_representatives[assigned, rep_position]
        neighbourhood = distance_values[:, representative_index] < radius
        nb = float(weights[neighbourhood].sum())
        na = float(assigned_weights.sum())
        sd = float(np.sum(assigned_distances * assigned_weights))
        md = float(sd / na) if na else 0.0
        dc = float(np.sum(total_center[assigned] * assigned_weights))
        v = float(np.mean(assigned_distances)) if assigned_distances.size else 0.0
        q = float((dc - sd) / dc * 100.0) if dc else 0.0
        stats_rows.append(
            {
                "na": na,
                "na(%)": (na / weights.sum()) * 100.0 if weights.sum() else 0.0,
                "nb": nb,
                "nb(%)": (nb / weights.sum()) * 100.0 if weights.sum() else 0.0,
                "SD": sd,
                "MD": md,
                "DC": dc,
                "V": v,
                "Q": q,
            }
        )

    stats = pd.DataFrame(stats_rows, index=[f"r{index + 1}" for index in range(len(representative_indices))])
    total_sd = float(np.sum(min_distance * weights))
    total_dc = float(np.sum(total_center * weights))
    total_row = pd.DataFrame(
        [
            {
                "na": float(weights.sum()),
                "na(%)": 100.0,
                "nb": float(np.sum(np.any(distances_to_representatives < radius, axis=1) * weights)),
                "nb(%)": float(np.sum(np.any(distances_to_representatives < radius, axis=1) * weights) / weights.sum() * 100.0)
                if weights.sum()
                else 0.0,
                "SD": total_sd,
                "MD": float(total_sd / weights.sum()) if weights.sum() else 0.0,
                "DC": total_dc,
                "V": float(np.mean(total_center)) if total_center.size else 0.0,
                "Q": float((total_dc - total_sd) / total_dc * 100.0) if total_dc else 0.0,
            }
        ],
        index=["Total"],
    )
    stats = pd.concat([stats, total_row])

    quality = float((total_dc - total_sd) / total_dc) if total_dc else 0.0
    return RepresentativeSequencesResult(
        dataset=dataset,
        indices=representative_indices,
        scores=pd.Series(score_values, index=dataset.sequence_ids, name="score"),
        distances=pd.DataFrame(distances_to_representatives, index=dataset.sequence_ids),
        groups=pd.Series(groups, index=dataset.sequence_ids, name="representative_group"),
        statistics=stats,
        quality=quality,
        criterion=criterion_name,
        radius=radius,
        coverage=resolved_coverage,
    )

