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

uv_test_options := "--isolated --no-dev --group test --python 3.12 --extra torch-cpu"
pytest_options := "-v --tb=short tests/"

test:
    uv run {{ uv_test_options }} \
        pytest -m "not legacy_tf" -rs {{ pytest_options }}

test-parity:
    uv run {{ uv_test_options }} --with tensorflow-cpu \
        pytest -m "legacy_tf" -rs {{ pytest_options }}

test-all:
    uv run {{ uv_test_options }} --with tensorflow-cpu pytest {{ pytest_options }}
