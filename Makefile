PACKAGE_NAME ?= diffusion_handwriting_generation
TEXT ?= "Hello World and goodbye"
SOURCE ?= "data/lineImages/a01/a01-000/a01-000u-01.tif"
EXP ?= "data/best_exp"
CONFIG ?= ""
CHECKPOINT ?= ""
OUTPUT ?= "prediction"

.PHONY: lint format test train install sync infer

install:
	uv sync

infer:
	PYTHONPATH="." uv run python $(PACKAGE_NAME)/inference.py \
		--prompt="$(TEXT)" \
		--source="$(SOURCE)" \
		--experiment_path="$(EXP)" \
		--config_path="$(CONFIG)" \
		--checkpoint_path="$(CHECKPOINT)" \
		--output="$(OUTPUT)"

train:
	PYTHONPATH="." uv run python $(PACKAGE_NAME)/train.py --config=$(CONFIG)

test:
	uv run pytest -s tests

format:
	@uv run ruff check $(PACKAGE_NAME) --select I --fix
	@uv run ruff format $(PACKAGE_NAME)
 
lint:
	@uv run mypy $(PACKAGE_NAME)
	@uv run ruff check $(PACKAGE_NAME) --fix