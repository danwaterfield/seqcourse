# Releasing SeqCourse

SeqCourse is still a `0.x` library, so each release should be treated as an
alpha milestone rather than a long-term compatibility promise.

## Before Tagging

Run the local checks:

```bash
. .venv/bin/activate
pytest -q
python -m build
python -m twine check dist/*
```

If you want to refresh the TraMineR fixture locally first:

```bash
Rscript scripts/export_traminer_goldens.R
SEQCOURSE_TRAMINER_GOLDEN=tests/goldens/traminer_reference.json pytest -q tests/test_parity_scaffolding.py
```

Confirm on GitHub that:

- the Python test matrix is green
- the parity job is green
- packaging checks are green

## Tagging

Create an annotated tag and push it:

```bash
git tag -a v0.1.0 -m "SeqCourse v0.1.0"
git push origin v0.1.0
```

## GitHub Release

Create a GitHub release from the tag and attach the built `dist/` artifacts if
you want a downloadable source and wheel snapshot before PyPI publishing.

## PyPI

SeqCourse is not wired for automated PyPI publishing yet. Once the PyPI
project exists and trusted publishing is configured, add that workflow step on
top of the existing package build check rather than publishing by hand.
