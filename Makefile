train:
	python diffusion_handwriting_generation/train.py --config=$(CONFIG)

test:
	pytest -s tests