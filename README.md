# Dopamine-PINN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20352595.svg)](https://doi.org/10.5281/zenodo.20352595)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Archived release: `v1.0-submission` — Medical & Biological Engineering &
Computing (MBEC) submission snapshot ([Zenodo record](https://doi.org/10.5281/zenodo.20352595)).

Physics-informed neural network solver for **2D** dopamine diffusion in
synaptic clefts, implemented in pure **JAX** (Flax NNX + Optax + jaxopt).
This repository accompanies the manuscript:

> Hackman, E., & Zhu, H. (2026). *When Inverse Physics-Informed Neural
> Networks Become Practically Identifiable: A Case Study in Synaptic
> Dopamine Transport.* Submitted to **Medical & Biological Engineering &
> Computing** (Springer, IFMBE).

The code reproduces every numeric result and figure in the paper. A
working DeepXDE/PyTorch reference implementation is preserved as
`dopamine_PINN_deepxde.{py,ipynb}` for cross-validation.

---

## Quickstart

### Colab (recommended)

1. Open [`dopamine_PINN.ipynb`](dopamine_PINN.ipynb) in Google Colab.
2. **Runtime → Change runtime type → GPU** (T4 is sufficient).
3. Run all cells (~10–15 min at full fidelity, ~3 min in QUICK mode).
4. Download `figures/metrics.json` and the three PNGs (the notebook does
   this automatically in the last cell).

### Local

```bash
pip install -r requirements.txt
python3 dopamine_PINN.py             # full run, ~10–15 min CPU
python3 dopamine_PINN.py --quick     # ~60-s sanity check
```

### Inject results into the manuscript

```bash
python3 validate_metrics.py             # dry run — preview substitutions
python3 validate_metrics.py --apply     # commit (writes .tex.bak backup)
python3 validate_metrics.py --restore   # revert
```

---

## Repository contents

| File | Purpose |
|---|---|
| `dopamine_PINN.py` | Canonical JAX solver — 2D forward + inverse problems |
| `dopamine_PINN.ipynb` | Colab notebook with the full experimental pipeline |
| `dopamine_PINN_deepxde.{py,ipynb}` | Preserved DeepXDE/PyTorch reference implementation (cross-validation) |
| `validate_metrics.py` | Helper for substituting numerical results into LaTeX manuscript placeholders |
| `refs.bib` | Vancouver-style bibliography for the accompanying manuscript |
| `figures/` | Generated PNGs + `metrics.json` + `variables.dat` from a complete reference run |
| `figures_tiff/` | 600-DPI TIFFs used in the journal submission |

The accompanying manuscript ("When Inverse Physics-Informed Neural Networks Become Practically Identifiable: A Case Study in Synaptic Dopamine Transport", Hackman & Zhu, 2026) is under review at *Medical & Biological Engineering & Computing* (Springer, IFMBE). The version of record will be made available via the journal's platform upon acceptance.

---

## The PDE

We model dopamine concentration $C(x, y, t)$ on a 2D square as a linear
reaction-diffusion equation:

$$\frac{\partial C}{\partial t} \;=\; D\,\nabla^2 C \;-\; k\, C, \qquad (x, y) \in [-L/2, L/2]^2,\; t \in [0, T]$$

with a radial Gaussian release pulse as the initial condition and
zero-flux (Neumann) boundary conditions on all four edges. Parameter
values used (Cragg & Rice 2004; Nicholson & Phillips 1981; Wiencke et al.
2020):

| Symbol | Value | Meaning |
|---|---|---|
| $D$ | 0.32 µm²/ms | Effective diffusion coefficient |
| $k$ | 0.05 1/ms | Linearised DAT reuptake rate |
| $L$ | 5.0 µm | Domain side length (volume-transmission scale) |
| $T$ | 20 ms | Post-release simulation window |
| $\sigma$ | 0.5 µm | Release pulse half-width |
| $C_0$ | 1.0 µM | Peak release concentration |

---

## What gets computed

### Forward problem
A PINN ($\mathcal{N}_\theta : (x, y, t) \to \hat{C}$, 4×64 fully-connected,
`tanh`, Glorot init, implemented as a Flax NNX module) is trained to
satisfy the PDE plus IC plus BC. Validated against:

- the closed-form 2D Gaussian + exponential-decay analytical solution
  (infinite-domain approximation, valid for $t < L^2/(2D)$);
- an explicit 5-point Laplacian finite-difference reference solver with
  Neumann mirror boundaries.

Reports `L2(PINN vs. analytical)` and `L2(PINN vs. FD)` as percentages.

### Inverse problem
$D$ and $k$ are exposed as `nnx.Param` attributes of an `InverseMLP`
module (log-parametrized to enforce positivity), with deliberately
offset initial guesses ($D_0 = 0.30$, $k_0 = 0.04$ vs. truth
$0.32$, $0.05$). The PINN sees 400 noisy synthetic observations
($\sigma = 2\%$ of peak concentration) sampled uniformly from the
$(x, y, t)$ space-time domain. Reports recovered $\hat{D}, \hat{k}$
and their relative errors.

---

## Reproducing the reported results

After running `dopamine_PINN.py` (or the notebook), the outputs land in
`figures/`:

| Output | Description |
|---|---|
| `forward_snapshots.png` | PINN vs analytical vs FD snapshots at three time slices |
| `forward_heatmap.png` | Space-time concentration field with pointwise error |
| `inverse_convergence.png` | $D$ and $k$ trajectories during inverse training |
| `noise_sweep.png` | Recovery vs observational noise level (5 levels) |
| `posterior.png` / `posterior_hmc.png` | Bayesian UQ via Laplace and HMC |
| `param_grid.png` | Recovery across the biological D-k range |
| `obs_density_scaling.png` | Recovery vs number of observations |
| `metrics.json` | All numerical results (forward L2, inverse D/k, multi-seed, noise sweep, etc.) |

---

## Hardware

- **GPU**: Colab T4 / A100 — full run ~5–10 min after JIT warmup.
- **CPU**: any modern x86 / Apple Silicon — full run ~10–15 min.
- **Memory**: ≤4 GB RAM peak (2D grid + collocation tensors).

Random seeds are fixed (`SEED = 1234`). Re-runs reproduce results to
within floating-point tolerance.

---

## Dependencies

- Python ≥ 3.10
- JAX ≥ 0.4.30 (CPU or `jax[cuda12]` for GPU)
- Flax ≥ 0.10 (NNX API), Optax ≥ 0.2.3, jaxopt ≥ 0.8
- NumPy, SciPy, Matplotlib

```bash
pip install -r requirements.txt
```

The DeepXDE/PyTorch reference implementation in `*_deepxde.{py,ipynb}`
needs `deepxde>=1.10` and `torch>=2.0`; see the commented block at the
bottom of `requirements.txt`.

---

## Citation

If you use this code, please cite the manuscript:

```bibtex
@article{Hackman2026Dopamine,
  author    = {Hackman, Emmanuel and Zhu, Huiqing},
  title     = {When Inverse Physics-Informed Neural Networks Become
               Practically Identifiable: A Case Study in Synaptic
               Dopamine Transport},
  journal   = {Medical \& Biological Engineering \& Computing},
  year      = {2026},
  note      = {Submitted}
}
```

---

## Author

**Emmanuel Hackman**
School of Mathematics and Natural Sciences
University of Southern Mississippi
`emmanuelhackman825@gmail.com`

---

## License

Released under the [MIT License](LICENSE). You are free to use, copy,
modify, distribute, and sublicense this code, provided the copyright
notice is retained.

---

## Reference papers cited in the manuscript

22 papers, organized into 5 topical folders under `Reference Papers/`:

- **01 – PINNs Core** (Raissi 2019, Lu 2021, Cai 2021, Raissi 2020)
- **02 – PINNs Biomedical & Inverse Problems** (Haghighat 2021,
  Rudy 2017, Sahli Costabal 2020, Yazdani 2020)
- **03 – Dopamine & Synaptic Diffusion** (Wiencke 2020, Rice & Cragg
  2008, Rodeberg 2017, Nicholson & Phillips 1981)
- **04 – Parkinson's Disease Computational** (GBD 2016, García 2017,
  Corti 2023, Pandya 2019)
- **05 – Reaction-Diffusion PDEs** (Wang 2021, Wu 2023, Raissi 2019)

See `Reference Papers/READING_GUIDE.md` for full bibliographic details
and reading order. `Reference Papers/download_papers.py` and
`download_pmc_pow.py` can re-download all 20 PDFs (including those
behind NCBI's JavaScript Proof-of-Work challenge).

---

## Acknowledgments

The author thanks the School of Mathematics and Natural Sciences at the
University of Southern Mississippi for computational resources, and
acknowledges the use of Claude (Anthropic) as a writing and editing
assistant during manuscript preparation (see the Artificial Intelligence
Disclosure section of the manuscript for the full statement).

---

## TODO before public release

- [x] Add `LICENSE` (MIT)
- [x] Add `requirements.txt` with version constraints
- [x] Push to GitHub at `github.com/Hachero98/dopamine-PINN-`
- [x] Link GitHub repo to Zenodo and mint a DOI
- [x] Replace placeholder DOI in the manuscript's Data
      Availability section
- [x] Tag a `v1.0-submission` release matching the MBEC manuscript submission
- [ ] After first install on the submission machine, run
      `pip freeze > requirements-frozen.txt` to lock exact versions
