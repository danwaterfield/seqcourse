# SeqCourse Roadmap

## Current State

SeqCourse has a usable alpha core:

- encoded `SequenceDataset` objects with wide and spell conversions
- core cost methods and distance methods for the main TraMineR-style workflow
- summaries, representative sequences, plotting, and a `compat` namespace
- Python test coverage across CPython 3.11, 3.12, and 3.13
- a green TraMineR parity workflow in GitHub Actions

What it does not have yet is release-grade speed or the broader TraMineR
surface area beyond the current core workflow.

## Immediate Priorities

1. Keep parity green while tightening the compatibility surface.
   The current parity contract now covers state ordering, optional `HAM`,
   optional weights, representative fixtures, and TraMineR-like handling of
   trailing missing values in compatibility mode.

2. Improve release polish.
   Packaging checks, release notes, examples, and a clearer public story around
   scope and limits are the next non-performance priorities.

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
- packaging checks in CI and a documented release checklist
- documented performance expectations for the pure Python backend
- a clearer statement of what is and is not yet equivalent to TraMineR
- at least one worked example beyond the README quick start
