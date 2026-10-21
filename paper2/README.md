# Paper 2 workbench

A clean, fast place to develop and check the counterfactual studies of paper 2 before
the long runs. It sits beside the existing code and does not change it: the paper-1
package `dopamine_pinn` supplies the FD solver, the network and the trainer, and
`dopamine_pinn/causal.py` with its Colab results stays as it is.

One pipeline serves every model:

| Model | Uptake under dose a | Drug parameter |
|---|---|---|
| `linear` | k (1 − η a) C | η |
| `mm_capacity` | V_max (1 − η a) C / (K_m + C) | η (transporter loss) |
| `mm_competitive` | V_max C / (K_m (1 + γ a) + C) | γ (competitive block) |

Each study: synthetic data at the training doses → reference fit → counterfactual
Cramér–Rao bound → staged PINN (stages 2–4) → held-out effect at the unseen dose, three
ways (network; PDE at the PINN's parameters; PDE at the reference's parameters) → saved.

## Running

From the repository root (Python 3.11 here, which has JAX, Flax and pytest):

```bash
python3.11 paper2/run.py --model linear --profile smoke --seeds 4001          # check it runs
python3.11 paper2/run.py --model linear --profile fast --seeds 4001-4005
python3.11 paper2/run.py --model mm_capacity --profile fast --seeds 4001-4005
python3.11 paper2/run.py --model linear --profile fast --w-res 30000 --tag wres3e4 --seeds 4009 4010
python3.11 paper2/run.py --model linear --profile fast --summary              # table + figure only
python3.11 -m pytest -q paper2/tests                                          # about 6 minutes on CPU
```

On Colab: open `paper2/colab.ipynb` from the `causal-linear` branch, set MODEL, PROFILE and
SEEDS, Run all. Results go to `MyDrive/paper2_runs/`.

Profiles (`cf/settings.py`):

| Profile | Collocation | Iterations per stage | Use |
|---|---|---|---|
| `smoke` | 300 | 200 + 20 | tests; numbers mean nothing |
| `fast` | 3,000 | 5,000 + 500 | quick checks and comparisons |
| `paper` | 30,000 | 20,000 + 2,000 | paper 1's settings, final runs |

Outputs: `runs/<model>/<profile>[_<tag>]/obs<seed>_net<seed>/` with `results.json` (every
number), `weights.pkl` (network weights, so fields and figures can be redrawn without
retraining) and three figures; `compare.png` in the profile folder shows PINN against
reference over all realizations. A finished study is skipped on rerun.

## Code (`cf/`)

| Module | Contents |
|---|---|
| `settings.py` | `Design` (doses, held-out dose, points, noise, release, record length) and training `Profile`s |
| `mechanisms.py` | the uptake models; **add a model here** (a subclass with its parameters and `uptake`) |
| `model.py` | FD forward model for any mechanism; observations (same draws as the earlier runs) |
| `reference.py` | staged least-squares reference with Laplace covariance; Fisher information |
| `effects.py` | τ, ATE(t), CE, GTE on the evaluation grid; the counterfactual bound |
| `pinn.py` | dose-aware network, loss for any mechanism, staged curriculum |
| `study.py` | `run_study`, `load_study`; diagnostics (data misfit vs reference, residual vs uptake) |
| `analysis.py` | paired PINN vs reference statistics over many realizations |
| `plots.py` | per-study and comparison figures from saved results |

Conventions: parameters are fitted on the log scale; realization seed s uses seed
s + 100 j for dose j, so seeds stay below 4101; errors of parameters are reported on the
log scale (comparable with the Laplace SD and the bound).

## What to check before a `paper` run

1. `fast` runs of each model finish and their reference sits at the bound.
2. The PINN-to-reference RMSE ratio, with its interval, on 5–10 realizations.
3. The diagnostics: PINN data misfit should not fall below the reference's, and the
   residual should be small against the uptake term. Paper-1 weights fail this for the
   dose-aware network (`causal-linear` Colab runs, Oct 2026); compare `--w-res` values
   with `fast` first.
