# Legacy files

Kept for the record; not used by the current results.

| File | What it was | Superseded by |
|---|---|---|
| `dopamine_PINN.py` | Script version of the first pipeline (T = 20 ms, old loss weights) | `dopamine_pinn/` package |
| `dopamine_PINN_deepxde.{py,ipynb}` | DeepXDE / PyTorch implementation of the first pipeline | `dopamine_pinn/` package |
| `validate_metrics.py` | Substituted numbers into the PLOS-era LaTeX manuscript | `dopamine_PINN_baseline.ipynb` -> `paper_numbers.json` |
| `Makefile` | Build pipeline of the PLOS-era manuscript | root `Makefile` |
| `figures_tiff/` | Figures 1-9 of the PLOS-era manuscript | manuscript figures built from the baseline results |
| `figures_v1_k0.05/` | Outputs of the first pipeline (k = 0.05, T = 20 ms) | `results/` |
