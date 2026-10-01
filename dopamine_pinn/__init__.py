"""dopamine_pinn: inverse PINNs and a finite-difference reference for synaptic dopamine transport.

Baseline code of Hackman & Zhu, "Inverse Physics-Informed Neural Networks for Synaptic
Dopamine Transport: Parameter Recovery at the Information Limit of Sparse, Noisy Data"
(Medical & Biological Engineering & Computing, submitted).

Modules
-------
config        model constants and the paper's default settings
solver        FD reference solver (JAX, differentiable; obstacle mask; any uptake law)
observations  synthetic observation designs (scattered, electrode-like, grid nodes)
reference     reference estimator with Laplace posterior; Levenberg-Marquardt baseline
pinn          network, PDE residuals, collocation, trainer (optional adaptive weights / RAD)
experiments   runners for the paper's experiments
benchmark     frozen benchmark datasets (generate / load)
"""
__version__ = '1.0.0'

import jax as _jax

_jax.config.update('jax_enable_x64', True)   # float64 is required to recover k

from . import config  # noqa: E402,F401
