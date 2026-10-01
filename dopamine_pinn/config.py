"""Model constants and default settings of the baseline paper (Hackman & Zhu, MBEC).

Every value here is the one used in the manuscript. Functions elsewhere in the package
take these as keyword defaults, so a later study changes a setting by passing an
argument, never by editing this file.
"""

# ---- Physics (Cragg & Rice 2004; see the manuscript, Table 1) --------------------
D_TRUE = 0.32       # effective diffusion coefficient, um^2/ms (0.763 / 1.54^2)
K_TRUE = 0.020      # linearised DAT reuptake rate, 1/ms (Vmax / Km = 4.1 uM/s / 0.21 uM)
VMAX_TRUE = 0.0041  # Michaelis-Menten Vmax, uM/ms (Cragg & Rice 2004)
KM_TRUE = 0.21      # Michaelis-Menten Km, uM (Cragg & Rice 2004)
D_FREE = 0.763      # free diffusion coefficient of dopamine, um^2/ms (Cragg & Rice 2004)
L = 5.0             # side of the square domain [-L/2, L/2]^2, um
T = 50.0            # record length, ms
SIGMA = 0.5         # width of the Gaussian release, um
C0 = 1.0            # peak release concentration, uM

# ---- Inverse problem ----------------------------------------------------------------
D_INIT, K_INIT = 0.30, 0.016   # offset initial guesses (-6.25%, -20%)
NOISE_SD = 0.02                # observation noise SD, uM (2% of C0)
T_WINDOW = (0.5, 0.9 * T)      # observation window, ms
N_OBS = 400

# ---- Network and training (final inverse setting) ----------------------------------
LAYERS = (3, 64, 64, 64, 64, 1)
ADAM_ITERS = 20_000
LBFGS_ITERS = 2_000
LR = 1e-3
N_CHUNK = 50
N_DOMAIN = 30_000
N_INITIAL = 400
N_BOUNDARY_INV = 2_000
N_BOUNDARY_FWD = 400
W_INVERSE = (3000.0, 1.0, 100.0, 5.0)   # residual, initial, boundary, data
W_FORWARD = (10.0, 1.0, 1.0, 0.0)
SEED = 1234                              # network and collocation seed

# ---- FD reference solver ------------------------------------------------------------
NX_FD = 81          # nodes per side (dx = 0.0625 um)
DT_STORE = 0.1      # ms between stored frames
D_CAP = 0.64        # static upper bound on D that fixes the time step (2 x D_TRUE)

# ---- Observation seeds used in the paper -------------------------------------------
SEEDS = {
    'tuning_residual_weight': [1234],
    'tuning_fresh_checks': [2001, 2002, 2003, 3001],
    'tuning_data_weight': [3001],
    'confirmation': [4001, 4002, 4003],
    'paper_canonical': [4001],
    'repeated_realizations': list(range(7001, 7021)),
    'truthfree_selection': [4001, 8001, 8002],
}
