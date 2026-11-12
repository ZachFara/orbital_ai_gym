# 0001 Python 3.10+, uv, a single pyproject

## Context

The project is a Gymnasium environment package with a NumPy physics core,
pygame rendering, and Stable-Baselines3 training scripts. The existing code
is Python; its consumers (Gymnasium, SB3, PyTorch) are Python. Environment
management on the machine is already `uv`.

## Decision

- Language: Python, `requires-python >= 3.10`; developed and CI-tested on
  3.12.
- Environment and installs: `uv` (`make setup`), one `.venv/` at the repo
  root. `uv.lock` is kept as the record of a working set; the package
  itself declares loose lower bounds in `pyproject.toml`.
- One `pyproject.toml` holds packaging metadata, optional extras
  (`render`, `train`, `dev`) and every tool's configuration. No `setup.py`,
  `requirements*.txt`, or per-tool rc files.

## Consequences

- Contributors need `uv` (or can `pip install -e ".[dev,train,render]"`).
- Tool configuration changes are reviewed in one file.
- The `dev` extra pins only lower bounds; CI resolves fresh, so a tool
  release that changes a rule shows up as a CI failure to be handled by a
  decision, not by loosening.
