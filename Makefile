VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
PYTEST ?= $(VENV)/bin/pytest
RUN_ID ?= $(shell date +%Y%m%d_%H%M%S)
EVIDENCE ?= evidence/$(RUN_ID)

.PHONY: all bootstrap doctor build test data train evaluate ablate stress bench docs demo acceptance verify release trace audit-corpus test-model test-formal search check-proofs depth-study clean

all: build test

bootstrap:
	uv venv $(VENV) --python 3.12
	uv pip install --python $(PYTHON) -e ".[dev]"

doctor:
	$(PYTHON) -m noema.doctor

build:
	uv pip install -e . --python $(PYTHON)
	cd lean && lake build || true

test:
	$(PYTEST) tests/ -v --timeout=120

test-model:
	$(PYTEST) tests/test_model*.py -v

test-formal:
	$(PYTEST) tests/test_formal*.py -v

data:
	$(PYTHON) -m noema.corpus.generate --output data/

audit-corpus:
	$(PYTHON) -m noema.corpus.audit --data data/

trace:
	$(PYTHON) -m noema.corpus.extract_traces --data data/

train:
	$(PYTHON) -m noema.training.run --config configs/train_24block.yaml

depth-study:
	$(PYTHON) -m noema.training.depth_study --output runs/depth_study/

search:
	$(PYTHON) -m noema.search.run --theorems data/eval_controlled.jsonl

check-proofs:
	$(PYTHON) -m noema.artifacts.check --proofs artifacts/proofs/

evaluate:
	$(PYTHON) -m noema.evaluation.run --checkpoint checkpoints/best.pt

ablate:
	$(PYTHON) -m noema.evaluation.ablations --output runs/ablations/

stress:
	$(PYTEST) tests/test_stress.py -v --timeout=300

bench:
	$(PYTHON) -m noema.benchmarks.run --output runs/benchmarks/

docs:
	$(PYTHON) scripts/render_docs.py

demo:
	$(PYTHON) -m noema.demo

acceptance:
	$(PYTHON) -m noema.acceptance --run-id $(RUN_ID) --out-dir evidence/$(RUN_ID)

verify:
	$(PYTHON) -m noema.verifier --evidence $(EVIDENCE)

release:
	$(PYTHON) -m noema.release --output dist/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
