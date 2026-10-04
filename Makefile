# Developer entry points. CI runs these same targets on Linux.
# On Windows without make, run the commands after each target with `uv run`.

IMAGE ?= yield-triage-agent:local

.PHONY: install lint typecheck test smoke eval hygiene docker data data-synthetic

install:
	uv sync --frozen --all-extras

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest --cov=yield_triage --cov-report=term-missing

smoke:
	uv run python scripts/smoke.py

# Runs the fixed-seed evaluation, then splices the generated table into README.md.
eval:
	uv run python eval/run_eval.py
	uv run python scripts/update_readme.py

hygiene:
	bash scripts/check_hygiene.sh

# Builds the image, then runs the same smoke test inside the container.
docker:
	docker build -t $(IMAGE) .
	docker run --rm $(IMAGE) python scripts/smoke.py

data:
	uv run python scripts/fetch_data.py

data-synthetic:
	uv run python scripts/fetch_data.py --synthetic
