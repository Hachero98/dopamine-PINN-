"""Regenerate the frozen benchmark datasets in benchmark/v1/ (CPU, a few minutes).

    python scripts/make_benchmark.py            # all datasets
    python scripts/make_benchmark.py linear_s4001 obstacles_s9201
"""
import sys
import time

from dopamine_pinn import benchmark

t0 = time.time()
benchmark.generate(names=sys.argv[1:] or None)
print(f'done in {time.time() - t0:.0f} s -> {benchmark.DEFAULT_DIR}')
