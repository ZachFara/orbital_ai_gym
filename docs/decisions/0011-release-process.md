# 0011 Releases: a tag builds, checks, publishes

## Context

The package is meant to be installed with `pip install orbital-gym` and
listed among Gymnasium's third-party environments. The owner publishes;
agents prepare (no external publishing without the owner).

## Decision

- The version lives in `pyproject.toml` only. A release is a tag `vX.Y.Z`
  on `master`; `.github/workflows/release.yml` builds the sdist and wheel
  with `uv build`, installs the wheel into a clean virtualenv and runs the
  smoke test against it, publishes a GitHub release with the archives and
  the reference moon checkpoint, and uploads to PyPI through trusted
  publishing (environment `pypi`). Nothing is uploaded from a laptop.
- Ids are registered bare and under the `orbital_gym/` namespace
  (Gymnasium's convention for third-party packages); both names are the
  same env forever (decision 0006).
- The reference checkpoint is committed under `docs/checkpoints/` so the
  README's benchmark reproduces from a clean clone and the release can
  attach it. Checkpoints are a few megabytes; a new one replaces the old
  file in the same PR that changes the README table.
- The expert state banks are cached in `$ORBITAL_GYM_CACHE`, else `cache/`
  in a checkout, else the user's cache directory. An installed package
  never writes into `site-packages`.
- Licence: MIT, the owner's name. (Chosen by the agent as the conventional
  default for a research environment package; the owner confirms before
  the first tag.)

## Consequences

- `pip install orbital-gym` is one PyPI registration away: the owner
  creates the project on PyPI with this repository and workflow as a
  trusted publisher, then pushes a tag.
- The Gymnasium third-party listing and any Hub upload (D2) are the
  owner's to send; the repository carries everything they link to.
