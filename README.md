# SeqCourse

SeqCourse is a Python sequence-analysis library for social trajectories, life
courses, and other categorical state sequences. The project aims to replace the
core state-sequence workflow of TraMineR with a Python-first API, careful test
coverage, and a backend architecture that can grow into native acceleration
later without changing the public interface.

## Why SeqCourse

- Python-native sequence objects instead of an R compatibility shim
- Explicit support for wide and spell representations
- A stable encoded-sequence core built around `SequenceDataset`
- A public API that can stay clean while a `compat` layer preserves
  TraMineR-style entry points where helpful
- CI and parity scaffolding for validating behaviour against TraMineR 2.2-13

## Current Features

- `SequenceDataset` with encoded `uint16` storage and metadata for alphabet,
  labels, colors, weights, time labels, missing values, and void states
- Format conversion with `from_wide()`, `from_spell()`, `to_wide()`, and
  `to_spell()`
- Cost generation for `CONSTANT`, `TRATE`, `FUTURE`, `INDELS`, and
  `INDELSLOG`
- Distance computation for `OM`, `HAM`, `DHD`, `LCS`, `LCP`, `RLCP`, `CHI2`,
  and `EUCLID`
- Summary functions for state distributions, state frequencies, transition
  rates, and mean time in state
- Representative-sequence extraction and Matplotlib-based plotting
- A backend boundary that currently uses a Python reference implementation and
  leaves room for later Rust acceleration

## Install

```bash
pip install seqcourse
```

For local development:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd

from seqcourse import (
    SequenceDataset,
    cost_matrix,
    distance_matrix,
    representative_sequences,
    state_distribution,
)

wide = pd.DataFrame(
    [
        ["A", "A", "B", "B"],
        ["A", "C", "C", "B"],
        ["B", "B", "B", "B"],
    ],
    columns=["t1", "t2", "t3", "t4"],
)

seqs = SequenceDataset.from_wide(wide)

costs = cost_matrix(seqs, method="TRATE")
distances = distance_matrix(seqs, method="OM", sm=costs)
distributions = state_distribution(seqs)
representatives = representative_sequences(seqs, criterion="freq")

print(costs.sm)
print(distances)
print(distributions.frequencies)
print(representatives.indices)
```

## TraMineR Compatibility

SeqCourse is not trying to recreate the TraMineR API verbatim, but it does ship
with a compatibility namespace for common workflows:

```python
from seqcourse import compat

seqs = compat.seqdef(wide)
costs = compat.seqcost(seqs, method="TRATE")
distances = compat.seqdist(seqs, method="OM", sm=costs)
```

The main public API stays Pythonic; `compat` exists to ease migration and
parity testing.

## Validation

The repository contains three layers of validation:

- unit and invariant tests for dataset handling, summaries, costs, distances,
  representatives, and plots
- benchmark scaffolding in `benchmarks/benchmark_distances.py`
- TraMineR parity scaffolding driven by `scripts/export_traminer_goldens.R`

The parity fixture currently covers:

- `TRATE` substitution costs and indel costs
- `OM`, `HAM`, and normalized `LCS` distances
- state distributions, entropy, transition rates, and mean time in state
- representative sequence indices and group assignments

The parity test is designed to run in CI when `Rscript` and TraMineR are
available. You can also generate and use the fixture locally:

```bash
Rscript scripts/export_traminer_goldens.R
pytest tests/test_parity_scaffolding.py
```

To point the parity test at a fixture stored somewhere else, set
`SEQCOURSE_TRAMINER_GOLDEN`:

```bash
SEQCOURSE_TRAMINER_GOLDEN=/tmp/traminer_reference.json pytest tests/test_parity_scaffolding.py
```

## Status

This is an early `0.x` library. The current focus is:

- tightening parity for the core TraMineR state-sequence workflow
- improving documentation and examples
- profiling hotspots before introducing a Rust backend

Still intentionally deferred:

- advanced OM variants such as `OMloc`, `OMspell`, and `OMstran`
- `TWED`, `NMS`, `NMSMST`, and `SVRspell`
- event-sequence and multidomain support
- clustering wrappers and the broader TraMineR-adjacent ecosystem

## Development

Run the tests:

```bash
pytest
```

Run the small benchmark harness:

```bash
python benchmarks/benchmark_distances.py --n-sequences 200 --length 32 --states 6
```

## License

SeqCourse is distributed under the GNU General Public License, version 2 or
later (`GPL-2.0-or-later`). See [LICENSE](LICENSE).
