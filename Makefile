.PHONY: lint format typecheck test check-all

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	mypy src/

test:
	pytest

check-all: lint format-check typecheck test
