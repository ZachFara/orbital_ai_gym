PY ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy
VULTURE ?= .venv/bin/vulture
LINT_IMPORTS ?= .venv/bin/lint-imports
PYTEST ?= .venv/bin/pytest

SRC = orbital_gym scripts tests tools

.PHONY: check fmt setup hooks

# The one check. Pre-commit and CI run exactly this target.
check:
	$(RUFF) format --check $(SRC)
	$(RUFF) check $(SRC)
	$(MYPY) $(SRC)
	$(VULTURE) orbital_gym scripts tools --min-confidence 80
	$(LINT_IMPORTS)
	$(PY) tools/check_layout.py
	SDL_VIDEODRIVER=dummy $(PYTEST) -q

# Apply the formatter and ruff's safe fixes.
fmt:
	$(RUFF) format $(SRC)
	$(RUFF) check --fix $(SRC)

# Create the virtualenv with every extra (uses uv).
setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev,train,render]"

# Install the git hooks (pre-commit runs `make check`, commit-msg checks the message).
hooks:
	.venv/bin/pre-commit install --hook-type pre-commit --hook-type commit-msg
