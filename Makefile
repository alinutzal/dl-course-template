# Convenience targets for OSC/Linux/macOS. On Windows, run the commands directly.
.PHONY: install test train seeds summary
install:
	pip install -e .
test:
	python -m pytest
train:
	python scripts/train.py --config configs/default.yaml
seeds:            # three seeds of the default experiment
	for s in 1 2 3; do python scripts/train.py --config configs/default.yaml seed=$$s; done
summary:
	python scripts/summarize_runs.py runs
