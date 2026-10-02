# Dopamine-PINN: common tasks
PYTHON ?= python3

.PHONY: install test benchmark check-benchmark causal causal-quick

install:            ## install the package (editable) with test dependencies
	$(PYTHON) -m pip install -e ".[test]"

test:               ## run the test suite (CPU, ~1 min)
	$(PYTHON) -m pytest -q

benchmark:          ## regenerate benchmark/v1 from the seeds (CPU, ~15 s)
	$(PYTHON) scripts/make_benchmark.py

check-benchmark:    ## verify the stored benchmark files (hashes, exact regeneration)
	$(PYTHON) -m pytest -q tests/test_benchmark.py

causal-quick:       ## causal study do(A=a) on the linear model, tiny schedule (CPU, ~3 min)
	$(PYTHON) scripts/run_causal_linear.py --quick

causal:             ## causal study do(A=a) on the linear model, paper settings (GPU advised)
	$(PYTHON) scripts/run_causal_linear.py
