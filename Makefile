.PHONY: install install-all test lint fmt typecheck check clean docs

install:            ## core + dev, no GDAL/torch needed
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

test:
	pytest -q --cov=vhagar --cov-report=term-missing

lint:
	ruff check src tests

fmt:
	ruff format src tests && ruff check --fix src tests

typecheck:
	mypy src/vhagar

check: lint test     ## what CI runs

clean:
	rm -rf .pytest_cache .ruff_cache .coverage build dist **/__pycache__
