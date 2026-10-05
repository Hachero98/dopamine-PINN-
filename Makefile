# Dopamine-PINN: common tasks
PYTHON ?= python3

.PHONY: install test benchmark check-benchmark

install:            ## install the package (editable) with test dependencies
	$(PYTHON) -m pip install -e ".[test]"

test:               ## run the test suite (CPU, ~1 min)
	$(PYTHON) -m pytest -q

benchmark:          ## regenerate benchmark/v1 from the seeds (CPU, ~15 s)
	$(PYTHON) scripts/make_benchmark.py

check-benchmark:    ## verify the stored benchmark files (hashes, exact regeneration)
	$(PYTHON) -m pytest -q tests/test_benchmark.py
