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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark core seqcourse distance methods.")
    parser.add_argument("--n-sequences", type=int, default=200)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = synthetic_sequences(args.n_sequences, args.length, args.states, args.seed)
    trate = cost_matrix(dataset, method="TRATE")
    methods = [
        ("HAM", {}),
        ("LCS", {}),
        ("OM", {"sm": trate}),
    ]

    for method, kwargs in methods:
        started = time.perf_counter()
        distance_matrix(dataset, method=method, **kwargs)
        elapsed = time.perf_counter() - started
        print(f"{method}: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
