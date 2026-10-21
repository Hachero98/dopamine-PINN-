# Results behind the manuscript

Outputs of the runs that produced the numbers and figures in the BSPC manuscript
(NVIDIA A100, Google Colab, `jax[cuda12]==0.10.2`, `flax==0.12.5`, unless marked CPU).
Section, table and figure numbers refer to that manuscript.

| File or folder | Produced by | Manuscript items |
|---|---|---|
| `manuscript_main_run/` | `dopamine_PINN.ipynb` (observation seed 4001) | Section 3.1 (forward), Table 1 upper rows, Supplementary S3 (fields, noise sweep, posterior, parameter grid, LM baseline, observation density, trajectories); `metrics.json` holds every number |
| `manuscript_extensions/E1_noise_draws.*` | `dopamine_PINN_extensions.ipynb`, experiment E1 | Table 1 lower rows, Fig. 2, Section 3.2 |
| `manuscript_extensions/E2_truthfree_wd.*` | `dopamine_PINN_extensions.ipynb`, experiment E2 | Section 2.3 (cross-validation), Supplementary S2 (Step 6) |
| `manuscript_extensions/measurement_model.*` | `dopamine_PINN_tuning.ipynb`, cell *Realistic electrode measurement* | Tables 3 and 4, Fig. 4, Section 3.4 |
| `manuscript_extensions/review_analyses.json` | `scripts/review_analyses.py` (CPU) | Table 2 (reference over 50 realizations), Table 3 (Cramér-Rao bounds), Fig. 3c,d |
| `exact_solution_check.json` | `scripts/exact_solution_check.py` (CPU) | Section 3.1, Table A.1 |

The electrode-design runs (Table 2, realizations 4001-4005) and the starting-value test
(Section 3.3) are in `design_map/electrode_design.json` and `design_map/start_test.json`.
GPU runs are not bit-reproducible; a rerun reproduces these numbers within the
run-to-run spread measured by `dopamine_PINN_baseline.ipynb`.
