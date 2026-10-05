"""PINN smoke tests: shapes, finite losses, a short training run lowers the loss, and
pinned parameters stay fixed."""
import numpy as np
import jax.numpy as jnp

from dopamine_pinn import config as cfg
from dopamine_pinn.pinn import Net, build_comps, sample_points, make_batch, train, TrainConfig, phys_linear, phys_mm, predict
from dopamine_pinn.solver import make_fd_solver, solve_at, linear_sink
from dopamine_pinn.observations import scattered


def _setup(ff=0):
    s = make_fd_solver(nx=41)
    obs = scattered(solve_at(s, cfg.D_TRUE, linear_sink(cfg.K_TRUE)), s, 4001, n_obs=50)
    net = Net(layers=(3, 16, 16, 1), ff_features=ff)
    pts = sample_points(n_domain=300, n_initial=50, n_boundary=40)
    return net, make_batch(pts, obs)


def test_paper_collocation_counts():
    pts = sample_points()
    assert pts['x_r'].shape == (cfg.N_DOMAIN,) and pts['x_i'].shape == (cfg.N_INITIAL,)
    assert pts['x_b'].shape == (cfg.N_BOUNDARY_INV,)


def test_training_lowers_loss():
    net, b = _setup()
    comps = build_comps(phys_linear, net)
    p0 = {'net': net.init(1), 'logD': jnp.log(cfg.D_INIT), 'logk': jnp.log(cfg.K_INIT)}
    l0 = float(jnp.dot(jnp.asarray(cfg.W_INVERSE), comps(p0, b)))
    p, info = train(p0, comps, b, TrainConfig.quick())
    assert np.isfinite(info['loss']) and info['loss'] < l0


def test_pinned_and_remedies_run():
    net, b = _setup(ff=4)
    pin = {'logD': jnp.log(cfg.D_TRUE)}
    comps = build_comps(phys_linear, net, pin=pin)
    p0 = {'net': net.init(2), 'logk': jnp.log(cfg.K_INIT)}
    pool = lambda s: tuple(sample_points(n_domain=600, n_initial=4, n_boundary=4, seed=s)[k] for k in ('x_r', 'y_r', 't_r'))
    p, info = train(p0, comps, b, TrainConfig.quick(anneal=True, rad=True), rad_pool=pool)
    assert 'logD' not in p and np.isfinite(info['loss'])
    assert all(w > 0 for w in info['weights'])
    C = predict(net, p, np.zeros(3), np.zeros(3), np.array([0.0, 1.0, 2.0]))
    assert C.shape == (3,) and np.all(np.isfinite(C))


def test_mm_physics_finite():
    net, b = _setup()
    comps = build_comps(phys_mm, net)
    p0 = {'net': net.init(3), 'logD': jnp.log(0.3), 'logVmax': jnp.log(0.004), 'logKm': jnp.log(0.2)}
    assert np.all(np.isfinite(np.asarray(comps(p0, b))))
