PACKAGE_NAME ?= diffusion_handwriting_generation

.PHONY: lint format test train install

install:
	uv sync

train:
	PYTHONPATH="." uv run python $(PACKAGE_NAME)/train.py --config=$(CONFIG)

test:
	uv run pytest -s tests

format:
	@uv run ruff check $(PACKAGE_NAME) --select I --fix
	@uv run ruff format $(PACKAGE_NAME) prepare_data.py
 
lint:
	@uv run mypy $(PACKAGE_NAME)
	@uv run ruff check $(PACKAGE_NAME) --fix