# Dopamine-PINN

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Inverse physics-informed neural networks (PINNs) and a finite-difference reference
estimator for synaptic dopamine transport, implemented in **JAX** (Flax NNX, Optax,
jaxopt). This repository is the baseline code and data of:

> Hackman, E., & Zhu, H. *Inverse Physics-Informed Neural Networks for Synaptic Dopamine
> Transport: Parameter Recovery at the Information Limit of Sparse, Noisy Data.*
> Submitted to *Medical & Biological Engineering & Computing*.

The Zenodo DOI of the release that accompanies the manuscript will be added here when
the release is archived. (DOI 10.5281/zenodo.20352595 archives an earlier version with
different parameters; do not use it to reproduce the current manuscript.)

---

## What is in the repository

| Path | Purpose |
|---|---|
| `dopamine_pinn/` | **The package.** Everything the results use: FD solver, observation designs, reference estimator, PINN, experiment runners, benchmark data I/O |
| `benchmark/v1/` | **Frozen benchmark data**: every observation set of the paper and its extensions, with truth and settings in `index.json` |
| `dopamine_PINN_baseline.ipynb` | **Reproduces every number in the manuscript** through the package, from the benchmark files, and measures run-to-run variation |
| `tests/` | Test suite (solver, reference estimator, PINN, benchmark integrity) |
| `scripts/make_benchmark.py` | Regenerates `benchmark/v1/` from the seeds |
| `dopamine_PINN.ipynb` | Original main pipeline (manuscript v1-v5 numbers; superseded by the baseline notebook) |
| `dopamine_PINN_tuning.ipynb` | Record of the loss-weight selection (Online Resource 3) and the electrode designs |
| `dopamine_PINN_extensions.ipynb` | Extension experiments: 20 realizations, truth-free weights, Michaelis-Menten, D(x, y), obstacles |
| `dopamine_PINN_remedies.ipynb` | Adaptive weights, adaptive sampling and Fourier features on the cases where the PINN failed |
| `results/` | **Outputs of the runs behind the manuscript** (`metrics.json`, figures, JSON of the extension experiments) |
| `design_map/` | Cramér-Rao experimental-design map for electrode-like recordings |
| `legacy/` | Superseded files, kept for the record (see `legacy/README.md`) |

---

## The model

Dopamine concentration $C(x, y, t)$ on the square $[-L/2, L/2]^2$:

$$\frac{\partial C}{\partial t} = \nabla\cdot(D\,\nabla C) - s(C),\qquad
C(x, y, 0) = C_0\,e^{-(x^2+y^2)/2\sigma^2},\qquad \partial_n C = 0 \text{ on the walls},$$

with linear uptake $s(C) = kC$ in the paper; Michaelis-Menten uptake
$s(C) = V_{\max}C/(K_m + C)$, a spatially varying $D(x, y)$ and impermeable obstacles are
available in the package.

| Symbol | Value | Meaning | Source |
|---|---|---|---|
| $D$ | 0.32 µm²/ms | effective diffusion coefficient, $0.763/1.54^2$ | Cragg & Rice 2004; Nicholson & Phillips 1981 |
| $k$ | 0.020 ms⁻¹ | linearised DAT uptake, $V_{\max}/K_m = 4.1/0.21$ s⁻¹ | Cragg & Rice 2004 |
| $L$ | 5 µm | domain side; neighbouring release site at 5 µm | Cragg & Rice 2004, Fig. 2 |
| $T$ | 50 ms | record length, about $1/k$ | modelling choice |
| $\sigma$ | 0.5 µm | width of the release | modelling choice |
| $C_0$ | 1 µM | peak concentration (a normalisation for the linear model) | — |

---

## Quick start

```bash
pip install -e ".[test]"        # or: pip install -r requirements.txt && pip install -e .
make test                        # about 1 min on a laptop CPU
```

```python
from dopamine_pinn import benchmark, experiments as ex
from dopamine_pinn.solver import make_fd_solver
from dopamine_pinn.reference import fit_linear
from dopamine_pinn.pinn import TrainConfig

obs, meta = benchmark.load('linear_s4001')       # the paper's canonical realization
ref = fit_linear(obs, make_fd_solver())           # reference estimator + Laplace SDs
res = ex.inverse(obs, TrainConfig())              # inverse PINN with the paper's settings
print(ref['D'], ref['k'], res['params'])
```

One inverse run takes about 3 minutes on an NVIDIA A100 (20,000 Adam iterations, then
L-BFGS, 30,000 collocation points).

### Reproducing the manuscript

The numbers in the submitted manuscript were produced by the notebooks below (Google
Colab, NVIDIA A100); their outputs are in `results/`.

| Manuscript item | Notebook | Output in `results/` |
|---|---|---|
| Forward accuracy (Table 2, Figs 2-3) | `dopamine_PINN.ipynb` | `manuscript_main_run/` |
| Inverse recovery on realization 4001 (Table 3, upper rows), posterior (Fig 6), noise sweep (Fig 5), LM baseline (Table 4) | `dopamine_PINN.ipynb` | `manuscript_main_run/metrics.json` |
| Parameter grid, observation density, ablation, trajectories (Online Resource 4) | `dopamine_PINN.ipynb` | `manuscript_main_run/` |
| 20 independent realizations (Table 3, lower rows; Fig 4; Section 3.3) | `dopamine_PINN_extensions.ipynb` (E1) | `manuscript_extensions/E1_noise_draws.*` |
| Data weight without the ground truth (Section 3.4) | `dopamine_PINN_extensions.ipynb` (E2) | `manuscript_extensions/E2_truthfree_wd.*` |
| Loss-weight selection record (Online Resource 3) | `dopamine_PINN_tuning.ipynb` | — |
| FD reference vs the exact bounded-domain solution (Online Resource 2, Table S2.1; Section 3.1) | `scripts/exact_solution_check.py` (NumPy only, about a minute) | `exact_solution_check.json` |

Open a notebook in Colab, choose an A100 runtime and run all cells. GPU runs are not
bit-reproducible, so a rerun agrees within the run-to-run spread (about 1-2 percentage
points in the error of $D$ for a single run).

**Consolidated route.** `dopamine_PINN_baseline.ipynb` regenerates the same experiments
through the `dopamine_pinn` package from the frozen benchmark data (the first cell
installs the pinned JAX stack and this repository), measures the run-to-run spread and
writes `paper_numbers.json`. It is the route to use for new work.

---

## Benchmark data (`benchmark/v1/`)

54 observation sets, about 1 MB. Each `.npz` holds `x_d`, `y_d`, `t_d` (locations, µm
and ms), `C_d` (noisy concentrations, µM) and `C_clean`; `index.json` records the true
parameters, design, noise, seed, solver settings and a SHA-256 per file.
`benchmark.load(name)` verifies the hash.

| Datasets | Use |
|---|---|
| `linear_s{seed}` | scattered design, 400 points, 2% noise; seeds 1234, 2001-2003, 3001 (tuning), 4001-4003 (confirmation; 4001 canonical), 7001-7020 (repeated realizations), 8001-8002 (truth-free selection) |
| `linear_noise{η}_s4001` | noise levels 0, 1, 2, 5, 10% at the same locations |
| `linear_grid_D{D}_k{k}_s4001` | parameter grid |
| `linear_n{N}_s4001` | 100 to 1600 observations |
| `electrode_*_s4001` | FSCV-like designs: one site at r = 1, 2, 2.5 µm, three sites, coarse sampling |
| `mm_T50_C1_s9001`, `mm_T500_C10_s9001` | Michaelis-Menten uptake, 50 ms at 1 µM and 500 ms at 10 µM |
| `dfield_s9101` | spatially varying D(x, y), early-time design |
| `obstacles_s9201` | extracellular space with eight cellular obstacles, free D = 0.763 µm²/ms |

The files are regenerated bit for bit by `scripts/make_benchmark.py` (checked by the
tests). They come from the package's JAX solver with frames stored every 0.1 ms; the
original NumPy pipeline stored frames every 0.02 ms, which changes the noise-free values
by at most 3 × 10⁻⁴ µM (1.5% of the noise SD), with identical locations and noise.

---

## Using this as a baseline

* **New physics** (another uptake law, a D(x, y) field, binding): write a residual
  function `physics(p, x, y, t, C, grad, lap)` (see `pinn.phys_mm`, `pinn.DField`), pass it
  to `experiments.inverse(..., physics=...)`, and give the FD solver the matching `sink`.
* **New observation designs**: `observations.scattered`, `electrode`, `grid_nodes`.
* **New training methods**: `pinn.TrainConfig` (adaptive weights, adaptive resampling)
  and `pinn.Net(ff_features=...)` (Fourier features).
* **Comparisons**: use the `benchmark/v1` files, and report differences against the
  run-to-run spread measured by the baseline notebook, so that an improvement is not
  confused with seed noise.

---

## Hardware and versions

Manuscript results: Google Colab, NVIDIA A100, `jax[cuda12]==0.10.2`, `flax==0.12.5`,
float64 throughout. GPU reductions are not bit-reproducible, so a rerun can differ in the
last digits; the baseline notebook measures this spread.

---

## Citation

```bibtex
@article{Hackman2026Dopamine,
  author  = {Hackman, Emmanuel and Zhu, Huiqing},
  title   = {Inverse Physics-Informed Neural Networks for Synaptic Dopamine Transport:
             Parameter Recovery at the Information Limit of Sparse, Noisy Data},
  journal = {Medical \& Biological Engineering \& Computing},
  year    = {2026},
  note    = {Submitted}
}
```

## Author

**Emmanuel Hackman**, School of Mathematics and Natural Sciences, University of Southern
Mississippi (`emmanuelhackman825@gmail.com`).

## License

[MIT](LICENSE).
