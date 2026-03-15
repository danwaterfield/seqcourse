from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from seqcourse import SequenceDataset, cost_matrix, distance_matrix


def synthetic_sequences(n_sequences: int, length: int, n_states: int, seed: int) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    alphabet = [f"S{index + 1}" for index in range(n_states)]
    data = rng.choice(alphabet, size=(n_sequences, length))
    columns = [f"t{index + 1}" for index in range(length)]
    return SequenceDataset.from_wide(pd.DataFrame(data, columns=columns))


def benchmark_call(repeats: int, func) -> tuple[float, float, float]:
    runs: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        func()
        runs.append(time.perf_counter() - started)
    return min(runs), sum(runs) / len(runs), max(runs)


def parse_methods(value: str) -> list[str]:
    methods = [method.strip().upper() for method in value.split(",") if method.strip()]
    if not methods:
        raise argparse.ArgumentTypeError("Provide at least one benchmark method.")
    supported = {"HAM", "LCS", "OM"}
    unsupported = sorted(set(methods) - supported)
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"Unsupported methods: {', '.join(unsupported)}. Supported methods: {', '.join(sorted(supported))}."
        )
    return methods


def distance_kwargs(method: str, trate) -> dict[str, object]:
    if method == "OM":
        return {"sm": trate}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark core seqcourse distance methods.")
    parser.add_argument("--n-sequences", type=int, default=200)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--methods", type=parse_methods, default=["HAM", "LCS", "OM"])
    args = parser.parse_args()

    dataset = synthetic_sequences(args.n_sequences, args.length, args.states, args.seed)
    for _ in range(max(args.warmup, 0)):
        cost_matrix(dataset, method="TRATE")

    trate_min, trate_mean, trate_max = benchmark_call(args.repeat, lambda: cost_matrix(dataset, method="TRATE"))
    trate = cost_matrix(dataset, method="TRATE")
    print(
        f"TRATE: min={trate_min:.3f}s mean={trate_mean:.3f}s max={trate_max:.3f}s "
        f"(n={args.n_sequences}, length={args.length}, states={args.states}, repeat={args.repeat})"
    )

    for method in args.methods:
        kwargs = distance_kwargs(method, trate)
        for _ in range(max(args.warmup, 0)):
            distance_matrix(dataset, method=method, **kwargs)
        result_min, result_mean, result_max = benchmark_call(
            args.repeat,
            lambda: distance_matrix(dataset, method=method, **kwargs),
        )
        print(f"{method}: min={result_min:.3f}s mean={result_mean:.3f}s max={result_max:.3f}s")


if __name__ == "__main__":
    main()
