default:
    @just --list

clean:
    @find . -type d -name "__pycache__" -exec rm -rf {} +
    @find . -type d -name ".pytest_cache" -exec rm -rf {} +
    @find . -type d -name ".ruff_cache" -exec rm -rf {} +

lint:
    @uv run ruff check --fix src/

format:
    @uv run ruff format src/

typecheck:
    @uv run ty check src/

test:
    @uv run --isolated --python 3.12 --extra torch-cpu pytest -m "not legacy_tf" -v --tb=short tests/

test-parity:
    @uv run --isolated --python 3.12 --with tensorflow-cpu --extra torch-cpu pytest -m legacy_tf -v --tb=short tests/

test-all:
    @uv run --isolated --python 3.12 --with tensorflow-cpu --extra torch-cpu pytest -v --tb=short tests/
