# SeqCourse Roadmap

## Current State

SeqCourse has a usable alpha core:

- encoded `SequenceDataset` objects with wide and spell conversions
- core cost methods and distance methods for the main TraMineR-style workflow
- summaries, representative sequences, plotting, and a `compat` namespace
- Python test coverage across CPython 3.11, 3.12, and 3.13

What it does not have yet is release-grade parity or release-grade speed.

## Immediate Priorities

1. Finish parity hardening for the core TraMineR workflow.
   Current blocker: the fixture export now succeeds, but the Python parity test
   still needs to normalize optional fixture fields correctly, especially
   absent weights on unweighted datasets.

2. Make parity CI trustworthy.
   The goal is a green parity workflow that generates fixtures, uploads them as
   artifacts, and verifies core outputs against TraMineR without manual steps.
   The fixture contract now includes dataset-level `with_missing` handling and
   optional `HAM` outputs for datasets where TraMineR defines that distance.

3. Keep benchmark reporting honest and repeatable.
   The benchmark harness now reports `min/mean/max` timings; the next step is to
   use it consistently before backend changes and before any release.

4. Start targeted acceleration only after profiling.
   The first candidates are `OM`, `HAM`, `DHD`, and `LCS`, with a Rust backend
   kept behind the existing backend interface.

## Deferred Scope

- advanced OM variants such as `OMloc`, `OMspell`, and `OMstran`
- `TWED`, `NMS`, `NMSMST`, and `SVRspell`
- event-sequence and multidomain support
- clustering wrappers and the wider TraMineR-adjacent ecosystem

## Release Shape

Before the first serious public release, SeqCourse should have:

- a green Python test matrix and green parity workflow
- documented performance expectations for the pure Python backend
- a clearer statement of what is and is not yet equivalent to TraMineR
- at least one worked example beyond the README quick start
