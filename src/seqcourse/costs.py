from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .analysis import mean_time_in_state, state_distribution, transition_rates
from .backends import get_backend
from .dataset import SequenceDataset
from .results import CostMatrixResult
from ._utils import normalize_method


def _ensure_dataset(seqdata: SequenceDataset | object) -> SequenceDataset:
    if isinstance(seqdata, SequenceDataset):
        return seqdata
    return SequenceDataset.from_wide(seqdata)


def _state_axis(dataset: SequenceDataset, with_missing: bool) -> tuple[np.ndarray, tuple[str, ...]]:
    states = dataset.alphabet
    codes = np.arange(1, dataset.n_states + 1, dtype=np.uint16)
    if with_missing and dataset.has_missing:
        states = states + (dataset.missing_state,)
        codes = np.append(codes, dataset.missing_code)
    return codes, states


def _cost_matrix_impl(
    seqdata: SequenceDataset | object,
    *,
    method: str,
    cval: float | None = None,
    with_missing: bool = False,
    miss_cost: float | None = None,
    time_varying: bool = False,
    weighted: bool = True,
    transition: str = "both",
    lag: int = 1,
    backend: str | None = None,
) -> CostMatrixResult:
    dataset = _ensure_dataset(seqdata)
    method_name = normalize_method(method)
    supported = {"CONSTANT", "TRATE", "FUTURE", "INDELS", "INDELSLOG"}
    if method_name not in supported:
        raise ValueError(f"Unsupported cost method {method!r}. Supported methods: {', '.join(sorted(supported))}.")

    if with_missing and not dataset.has_missing:
        with_missing = False

    codes, states = _state_axis(dataset, with_missing=with_missing)
    n_states = len(states)
    if cval is None:
        cval = 4.0 if time_varying and method_name == "TRATE" and transition == "both" else 2.0
    if miss_cost is None:
        miss_cost = cval

    if method_name == "CONSTANT":
        if time_varying:
            costs = np.full((n_states, n_states, dataset.n_positions), float(cval), dtype=float)
            for position in range(dataset.n_positions):
                np.fill_diagonal(costs[:, :, position], 0.0)
        else:
            costs = np.full((n_states, n_states), float(cval), dtype=float)
            np.fill_diagonal(costs, 0.0)
        return CostMatrixResult(costs, indel=1.0, states=states, method=method_name, time_varying=time_varying)

    if method_name == "FUTURE":
        if time_varying:
            raise ValueError("The FUTURE cost method does not support time-varying output.")
        transitions = transition_rates(dataset, time_varying=False, weighted=weighted, lag=lag, with_missing=with_missing)
        column_sums = transitions.sum(axis=0)
        inverse = np.zeros_like(column_sums, dtype=float)
        nonzero = column_sums > 0
        inverse[nonzero] = 1.0 / column_sums[nonzero]
        costs = np.zeros((n_states, n_states), dtype=float)
        for left in range(n_states):
            for right in range(left + 1, n_states):
                distance = np.sqrt(np.sum(inverse * (transitions[left] - transitions[right]) ** 2))
                costs[left, right] = distance
                costs[right, left] = distance
        return CostMatrixResult(costs, indel=0.5 * float(costs.max(initial=0.0)), states=states, method=method_name)

    if method_name in {"INDELS", "INDELSLOG"}:
        if time_varying:
            distribution = state_distribution(dataset, weighted=weighted, with_missing=with_missing)
            proportions = distribution.frequencies.to_numpy(dtype=float)
            transformed = proportions.copy()
            transformed[transformed == 0] = 1.0
            if method_name == "INDELSLOG":
                transformed = np.log(2.0 / (1.0 + transformed))
            else:
                transformed = 1.0 / transformed
            costs = np.zeros((n_states, n_states, dataset.n_positions), dtype=float)
            for position in range(dataset.n_positions):
                for left in range(n_states):
                    for right in range(n_states):
                        if left == right:
                            continue
                        costs[left, right, position] = transformed[left, position] + transformed[right, position]
            return CostMatrixResult(costs, indel=transformed, states=states, method=method_name, time_varying=True)

        weights = dataset.weights if weighted else np.ones(dataset.n_sequences, dtype=float)
        counts = np.zeros(n_states, dtype=float)
        for index, code in enumerate(codes):
            counts[index] = np.sum((dataset.data == code) * weights[:, None])
        proportions = counts / counts.sum() if counts.sum() else np.zeros_like(counts)
        proportions[proportions == 0] = 1.0
        if method_name == "INDELSLOG":
            indel = np.log(2.0 / (1.0 + proportions))
        else:
            indel = 1.0 / proportions
        costs = np.zeros((n_states, n_states), dtype=float)
        for left in range(n_states):
            for right in range(n_states):
                if left == right:
                    continue
                costs[left, right] = indel[left] + indel[right]
        return CostMatrixResult(costs, indel=indel, states=states, method=method_name)

    transitions = transition_rates(dataset, time_varying=time_varying, weighted=weighted, lag=lag, with_missing=with_missing)
    if time_varying:
        costs = np.zeros((n_states, n_states, dataset.n_positions), dtype=float)

        def previous_cost(position: int, left: int, right: int) -> float:
            if position - lag < 0:
                return 0.0
            return -(transitions[left, right, position - lag] + transitions[right, left, position - lag])

        def next_cost(position: int, left: int, right: int) -> float:
            if position >= transitions.shape[2]:
                return 0.0
            return -(transitions[left, right, position] + transitions[right, left, position])

        for position in range(dataset.n_positions):
            at_start = position - lag < 0
            at_end = position >= transitions.shape[2]
            for left in range(n_states):
                for right in range(left + 1, n_states):
                    previous_component = previous_cost(position, left, right)
                    next_component = next_cost(position, left, right)
                    if transition == "previous":
                        cost = cval + previous_component
                    elif transition == "next":
                        cost = cval + next_component
                    else:
                        if not at_start and not at_end:
                            cost = cval + previous_component + next_component
                        else:
                            cost = cval + 2.0 * (previous_component + next_component)
                    cost = max(0.0, float(cost))
                    costs[left, right, position] = cost
                    costs[right, left, position] = cost
        return CostMatrixResult(costs, indel=0.5 * float(costs.max(initial=0.0)), states=states, method=method_name, time_varying=True)

    costs = np.zeros((n_states, n_states), dtype=float)
    for left in range(n_states):
        for right in range(left + 1, n_states):
            cost = max(0.0, float(cval - transitions[left, right] - transitions[right, left]))
            costs[left, right] = cost
            costs[right, left] = cost
    return CostMatrixResult(costs, indel=0.5 * float(costs.max(initial=0.0)), states=states, method=method_name)


def cost_matrix(
    seqdata: SequenceDataset | object,
    *,
    method: str,
    cval: float | None = None,
    with_missing: bool = False,
    miss_cost: float | None = None,
    time_varying: bool = False,
    weighted: bool = True,
    transition: str = "both",
    lag: int = 1,
    backend: str | None = None,
) -> CostMatrixResult:
    engine = get_backend(backend)
    return engine.compute_cost_matrix(
        seqdata,
        method=method,
        cval=cval,
        with_missing=with_missing,
        miss_cost=miss_cost,
        time_varying=time_varying,
        weighted=weighted,
        transition=transition,
        lag=lag,
    )


def seqcost(seqdata: SequenceDataset | object, *args: object, **kwargs: object) -> CostMatrixResult:
    return cost_matrix(seqdata, *args, **kwargs)


def seqsubm(seqdata: SequenceDataset | object, *args: object, **kwargs: object) -> np.ndarray:
    return cost_matrix(seqdata, *args, **kwargs).sm

