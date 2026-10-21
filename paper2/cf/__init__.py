"""cf: mechanistic counterfactual inference for dopamine transport (paper 2 workbench).

One pipeline for every uptake model; a model is a small class in `mechanisms`.

settings    Design (doses, held-out dose, noise, release) and training Profiles (smoke/fast/paper)
mechanisms  uptake laws and drug pathways: linear, mm_capacity, mm_competitive
model       FD forward model for any mechanism; synthetic observations
reference   least-squares reference (staged) and Fisher information
effects     effect summaries (tau, ATE, CE, GTE) and the counterfactual Cramer-Rao bound
pinn        dose-aware network and the staged curriculum
study       run_study / load_study: one complete study, saved to disk
analysis    tables over many studies: PINN vs reference, against the bound
plots       figures from saved studies
"""
try:
    import dopamine_pinn  # noqa: F401  (paper 1 package; also enables float64 in JAX)
except ModuleNotFoundError:   # not installed: use the copy in this repository
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import dopamine_pinn  # noqa: F401

from .settings import Design, Profile, PROFILES  # noqa: E402,F401
from .mechanisms import MECHANISMS  # noqa: E402,F401
from .study import run_study, load_study  # noqa: E402,F401
