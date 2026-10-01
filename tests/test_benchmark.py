"""Benchmark files are intact and regenerate bit-for-bit from their seeds."""
import numpy as np
import pytest

from dopamine_pinn import benchmark

pytestmark = pytest.mark.skipif(not (benchmark.DEFAULT_DIR / 'index.json').exists(),
                                reason='benchmark/v1 not generated')


def test_all_files_load_and_match_hash():
    for name in benchmark.names():
        obs, meta = benchmark.load(name)
        assert len(obs['C_d']) == meta['n']
        assert np.all(np.isfinite(obs['C_d']))


@pytest.mark.parametrize('name', ['linear_s4001', 'linear_s7001', 'electrode_S3_s4001'])
def test_regenerates_exactly(name):
    obs, _ = benchmark.load(name)
    build, _ = benchmark.specs()[name]
    new = build()
    for k in obs:
        assert np.array_equal(np.asarray(new[k]), obs[k]), k
