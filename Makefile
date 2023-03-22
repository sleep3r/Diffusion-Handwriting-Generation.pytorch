PACKAGE_NAME ?= diffusion_handwriting_generation

.PHONY: lint format test train install

install:
	pip install -e .

train:
	PYTHONPATH="." python $(PACKAGE_NAME)/train.py --config=$(CONFIG)

test:
	pytest -s tests

format:
	@isort $(PACKAGE_NAME)
	@black $(PACKAGE_NAME)
