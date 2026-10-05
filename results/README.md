# Results behind the manuscript

Outputs of the runs that produced the numbers and figures in the manuscript
(NVIDIA A100, Google Colab, `jax[cuda12]==0.10.2`, `flax==0.12.5`).

| Folder | Produced by | Manuscript items |
|---|---|---|
| `manuscript_main_run/` | `dopamine_PINN.ipynb` (observation seed 4001) | Table 2 (forward), Table 3 upper rows, Table 4 (LM baseline), Figs 2, 3, 5, 6, parameter grid, observation-density sweep, ablation and trajectories (Online Resource 4); `metrics.json` holds every number |
| `manuscript_extensions/E1_noise_draws.*` | `dopamine_PINN_extensions.ipynb`, experiment E1 | Table 3 lower rows, Fig 4, Section 3.3, Online Resource 4 (S4.5) |
| `manuscript_extensions/E2_truthfree_wd.*` | `dopamine_PINN_extensions.ipynb`, experiment E2 | Section 3.4, Online Resource 3 (Step 6) |
| `exact_solution_check.json` | `scripts/exact_solution_check.py` | Online Resource 2, Table S2.1, and Section 3.1 (FD within 0.06% of the exact bounded solution) |

The loss-weight selection record (Online Resource 3, Steps 1-5) comes from
`dopamine_PINN_tuning.ipynb`. GPU runs are not bit-reproducible; a rerun reproduces
these numbers within the run-to-run spread measured by `dopamine_PINN_baseline.ipynb`.
