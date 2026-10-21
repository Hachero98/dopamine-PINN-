"""Paper 2, tasks A1 and T2: reference estimator over many realizations against the
counterfactual Cramer-Rao bound, on the exact linear dose model.

For each observation seed (training doses 0, 0.25, 0.75, 1; 400 scattered points per
dose; 2% noise; the draws of the PINN studies), fit (D, k, eta) by least squares with
the exact model, then the held-out effect at A = 0.5: GTE, ATE(t) and tau(x, y, t).
Laplace (delta-method) intervals give coverage; the counterfactual bound at the truth
(per design) gives the smallest achievable error. Realizations 4001-4020 are the ones
the PINN runs on Colab; if their results are found (--pinn-dir), they are added for
comparison.

    python scripts/counterfactual_A1.py                      # seeds 4001-4050, ~1 min CPU
    python scripts/counterfactual_A1.py --seeds 4001 4002 --out /tmp/a1

Writes <out>/A1_reference_mc.json and <out>/A1_reference_mc.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from dopamine_pinn import counterfactual as cf  # noqa: E402
from dopamine_pinn.causal import PARAMS, TRUE, A_TRAIN, A_HELDOUT  # noqa: E402

Z = 1.96


def pinn_results(pinn_dir):
    """Stage-4/5 PINN numbers for obs*_net1234 studies, keyed by observation seed."""
    out = {}
    for p in sorted(Path(pinn_dir).glob('obs*_net1234/results.json')):
        r = json.loads(p.read_text())
        s5 = r['stage5']
        out[int(r['design']['obs_seed'])] = {
            'err_pct': r['pinn']['stage4']['err_pct'],
            'GTE_err_pct_pde': s5['pinn_calibrated_pde']['GTE_err_pct'],
            'GTE_err_pct_net': s5['pinn_network']['GTE_err_pct'],
            'tau_L2_pct_pde': s5['pinn_calibrated_pde']['E_L2_tau_pct'],
            'tau_L2_pct_net': s5['pinn_network']['E_L2_tau_pct'],
            'ref_fd_err_pct': r['reference']['stage4']['err_pct']}
    return out


def run(seeds, a=A_HELDOUT, doses=np.linspace(0.0, 1.0, 21)):
    grid = cf.EffectGrid()
    true = grid.gradients(a)                          # values and Jacobians at the truth
    gte0, ate0, tau0 = true['GTE'], true['ATE'], true['tau']
    n_tau, n_ate = np.linalg.norm(tau0), np.linalg.norm(ate0)
    pos = grid.ts > 0                                 # ATE(0) = 0 exactly, with zero SD
    grad_at = jax.jit(lambda th: (grid.gte(th, a), jax.jacfwd(grid.gte)(th, a),
                                  grid.ate(th, a), jax.jacfwd(grid.ate)(th, a),
                                  grid.tau(th, a), jax.jacfwd(grid.tau)(th, a)))
    g_dose = np.stack([np.asarray(jax.jacfwd(grid.gte)(jnp.asarray(cf.TH_TRUE), d)) for d in doses])
    gte_dose = np.array([float(grid.gte(jnp.asarray(cf.TH_TRUE), d)) for d in doses])

    rows, var_dose, ate_hat, var_ate = [], [], [], []
    for s in seeds:
        obs = cf.exact_observations(s)
        F = cf.fisher(obs)
        crb = cf.counterfactual_crlb(F, true)
        Fi = np.linalg.inv(F)
        var_dose.append(np.einsum('di,ij,dj->d', g_dose, Fi, g_dose))
        var_ate.append(crb['sd_ATE'] ** 2)
        fit = cf.fit_exact(obs)
        th, cov = fit['theta'], fit['cov']
        gte, dgte, ate, date, tau, dtau = (np.asarray(v) for v in grad_at(jnp.asarray(th)))
        sd_log = np.sqrt(np.diag(cov))
        sd_gte = float(cf.delta_sd(cov, dgte))
        sd_ate = cf.delta_sd(cov, date)
        sd_tau = cf.delta_sd(cov, dtau)
        ate_hat.append(ate)
        rows.append({
            'seed': int(s), 'chi2_per_obs': fit['chi2'] / obs['C_d'].size, 'at_bound': fit['at_bound'],
            'params': fit['params'],
            'err_pct': {p: 100 * (fit['params'][p] - TRUE[p]) / TRUE[p] for p in PARAMS},
            'err_log': {p: float(th[i] - cf.TH_TRUE[i]) for i, p in enumerate(PARAMS)},
            'sd_log': dict(zip(PARAMS, sd_log.tolist())),
            'covers': {p: bool(abs(th[i] - cf.TH_TRUE[i]) <= Z * sd_log[i]) for i, p in enumerate(PARAMS)},
            'crb_sd_log': crb['sd_log_params'],
            'GTE': float(gte), 'GTE_err_pct': 100 * (float(gte) - gte0) / gte0,
            'GTE_sd_pct': 100 * sd_gte / gte0, 'GTE_crb_pct': 100 * crb['sd_GTE'] / gte0,
            'GTE_covers': bool(abs(float(gte) - gte0) <= Z * sd_gte),
            'ATE_L2_pct': 100 * float(np.linalg.norm(ate - ate0) / n_ate),
            'ATE_L2_crb_pct': 100 * crb['rmse_L2_ATE'] / n_ate,
            'ATE_pointwise_coverage': float(np.mean(np.abs(ate - ate0)[pos] <= Z * sd_ate[pos])),
            'tau_L2_pct': 100 * float(np.linalg.norm(tau - tau0) / n_tau),
            'tau_L2_crb_pct': 100 * crb['rmse_L2_tau'] / n_tau,
            'tau_pointwise_coverage': float(np.mean((np.abs(tau - tau0) <= Z * sd_tau)[pos])),
        })

    var_dose = np.array(var_dose)
    S = cf.summarise
    summary = {
        'params_pct': {p: {**S([r['err_pct'][p] for r in rows]),
                           'log': S([100 * r['err_log'][p] for r in rows], [100 * r['sd_log'][p] for r in rows],
                                    [r['covers'][p] for r in rows]),
                           'rms_crb_pct': float(100 * np.sqrt(np.mean([r['crb_sd_log'][p] ** 2 for r in rows])))}
                       for p in PARAMS},
        'GTE_pct': {**S([r['GTE_err_pct'] for r in rows], [r['GTE_sd_pct'] for r in rows],
                        [r['GTE_covers'] for r in rows]),
                    'rms_crb_pct': float(np.sqrt(np.mean([r['GTE_crb_pct'] ** 2 for r in rows])))},
        'ATE_L2_pct': {'mean': float(np.mean([r['ATE_L2_pct'] for r in rows])),
                       'rms': float(np.sqrt(np.mean([r['ATE_L2_pct'] ** 2 for r in rows]))),
                       'rms_crb': float(np.sqrt(np.mean([r['ATE_L2_crb_pct'] ** 2 for r in rows]))),
                       'mean_pointwise_coverage': float(np.mean([r['ATE_pointwise_coverage'] for r in rows]))},
        'tau_L2_pct': {'mean': float(np.mean([r['tau_L2_pct'] for r in rows])),
                       'rms': float(np.sqrt(np.mean([r['tau_L2_pct'] ** 2 for r in rows]))),
                       'rms_crb': float(np.sqrt(np.mean([r['tau_L2_crb_pct'] ** 2 for r in rows]))),
                       'mean_pointwise_coverage': float(np.mean([r['tau_pointwise_coverage'] for r in rows]))},
        'n_at_bound': int(sum(r['at_bound'] for r in rows)),
    }
    ate_hat = np.array(ate_hat)
    curves = {'ts': grid.ts.tolist(), 'ATE_true': ate0.tolist(), 'ATE_mc_mean': ate_hat.mean(0).tolist(),
              'ATE_mc_sd': ate_hat.std(0, ddof=1).tolist(),
              'ATE_crb_sd_rms': np.sqrt(np.mean(var_ate, 0)).tolist(),
              'doses': doses.tolist(), 'GTE_true_dose': gte_dose.tolist(),
              'GTE_crb_pct_dose': (100 * np.sqrt(var_dose.mean(0)) / np.where(gte_dose > 0, gte_dose, np.nan)).tolist(),
              'GTE_crb_sd_dose': np.sqrt(var_dose.mean(0)).tolist()}
    design = {'A_train': list(A_TRAIN), 'A_heldout': a, 'seeds': [int(s) for s in seeds],
              'model': 'exact image series (truth and fitted model)', 'interval': f'+-{Z} Laplace SD',
              'GTE_true': gte0, 'GTE_closed_form': cf.gte_closed_form(a),
              'dGTE_dlogD_over_dGTE_dlogk': float(true['dGTE'][0] / true['dGTE'][1])}
    return {'design': design, 'summary': summary, 'curves': curves, 'realizations': rows}


def plot(res, pinn, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, sm, cv = res['realizations'], res['summary'], res['curves']
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    a = ax[0, 0]
    for i, p in enumerate(PARAMS):
        e = [100 * r['err_log'][p] for r in rows]
        a.scatter(np.full(len(e), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(e)), e, s=10, c='C0',
                  label='reference' if i == 0 else None)
        c = sm['params_pct'][p]['rms_crb_pct']
        a.fill_between([i - 0.3, i + 0.3], -Z * c, Z * c, color='0.85', zorder=0, label='±1.96 CRB' if i == 0 else None)
        pe = [100 * np.log1p(v['err_pct'][p] / 100) for v in pinn.values()]
        if pe:
            a.scatter(np.full(len(pe), i + 0.32), pe, marker='x', c='C3', s=25, label='PINN' if i == 0 else None)
    a.set_xticks(range(3))
    a.set_xticklabels(PARAMS)
    a.axhline(0, color='k', lw=0.6)
    a.set_ylabel('error of log-parameter (%)')
    a.set_title('Parameters, training doses {0, 0.25, 0.75, 1}')
    a.legend(fontsize=8)

    a = ax[0, 1]
    e = np.array([r['GTE_err_pct'] for r in rows])
    c = sm['GTE_pct']['rms_crb_pct']
    a.hist(e, bins=15, density=True, color='C0', alpha=0.6, label='reference')
    xs = np.linspace(-4 * c, 4 * c, 200)
    a.plot(xs, np.exp(-xs**2 / (2 * c**2)) / (c * np.sqrt(2 * np.pi)), 'k-', label=f'N(0, CRB²), CRB {c:.1f}%')
    for j, v in enumerate(pinn.values()):
        a.axvline(v['GTE_err_pct_pde'], color='C3', lw=0.8, alpha=0.7, label='PINN (PDE at λ̂)' if j == 0 else None)
    a.set_xlabel(f'GTE error at A = {res["design"]["A_heldout"]} (%)')
    a.set_title('Held-out global effect')
    a.legend(fontsize=8)

    a = ax[0, 2]
    ts, m = np.array(cv['ts']), np.array(cv['ATE_true'])
    a.plot(ts, m, 'k-', label='truth')
    a.plot(ts, cv['ATE_mc_mean'], 'C0--', label='reference, mean')
    sd_mc, sd_crb = np.array(cv['ATE_mc_sd']), np.array(cv['ATE_crb_sd_rms'])
    a.fill_between(ts, m - Z * sd_crb, m + Z * sd_crb, color='0.85', label='±1.96 CRB')
    a.plot(ts, m - Z * sd_mc, 'C0:', lw=1, label='±1.96 SD over realizations')
    a.plot(ts, m + Z * sd_mc, 'C0:', lw=1)
    a.set_xlabel('t (ms)')
    a.set_ylabel('ATE(t) (µM)')
    a.set_title('Domain-averaged effect')
    a.legend(fontsize=8)

    a = ax[1, 0]
    items = [(p, sm['params_pct'][p]['log']) for p in PARAMS] + [('GTE', sm['GTE_pct'])]
    for i, (name, s) in enumerate(items):
        lo, hi = s['coverage_ci95']
        a.errorbar(i, s['coverage'], yerr=[[s['coverage'] - lo], [hi - s['coverage']]], fmt='o', c='C0', capsize=4)
    a.plot([len(items)], [sm['ATE_L2_pct']['mean_pointwise_coverage']], 's', c='C2', label='pointwise, mean')
    a.plot([len(items) + 1], [sm['tau_L2_pct']['mean_pointwise_coverage']], 's', c='C2')
    a.axhline(0.95, color='k', ls='--', lw=0.8)
    a.set_xticks(range(len(items) + 2))
    a.set_xticklabels([n for n, _ in items] + ['ATE(t)', 'τ(x,y,t)'])
    a.set_ylim(0.6, 1.02)
    a.set_ylabel('coverage of 95% Laplace interval')
    a.set_title(f'Coverage, {len(rows)} realizations (Clopper-Pearson 95%)')
    a.legend(fontsize=8, loc='lower left')

    a = ax[1, 1]
    d = np.array(cv['doses'])
    a.plot(d[1:], np.array(cv['GTE_crb_pct_dose'])[1:], 'k-')
    for at in A_TRAIN:
        a.axvline(at, color='0.8', lw=0.8, zorder=0)
    a.axvline(res['design']['A_heldout'], color='orange', ls='--', lw=1, label='held-out dose')
    a.set_xlabel('dose A')
    a.set_ylabel('CRB of GTE_A (% of GTE_A)')
    a.set_title('Counterfactual bound across doses (T3 input)')
    a.legend(fontsize=8)

    a = ax[1, 2]
    a.scatter([r['tau_L2_crb_pct'] for r in rows], [r['tau_L2_pct'] for r in rows], s=12, c='C0', label='reference')
    by_seed = {r['seed']: r for r in rows}
    pts = [(by_seed[s]['tau_L2_crb_pct'], v['tau_L2_pct_pde']) for s, v in pinn.items() if s in by_seed]
    if pts:
        a.scatter(*zip(*pts), marker='x', c='C3', s=30, label='PINN (PDE at λ̂)')
    lim = a.get_xlim()
    a.axhline(sm['tau_L2_pct']['rms_crb'], color='k', ls='--', lw=0.8, label='RMS bound')
    a.set_xlim(lim)
    a.set_xlabel('bound on RMS L2 error of τ (%), this design')
    a.set_ylabel('L2 error of τ (%)')
    a.set_title('Held-out effect field')
    a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(4001, 4051)))
    ap.add_argument('--out', default='results/paper2_linear')
    ap.add_argument('--pinn-dir', default='dopamine_PINN_causal')
    args = ap.parse_args()
    assert max(args.seeds) < 4101, 'realization seeds would collide with dose seeds'

    res = run(args.seeds)
    pinn = pinn_results(args.pinn_dir) if Path(args.pinn_dir).exists() else {}
    if pinn:
        res['pinn'] = {str(k): v for k, v in pinn.items()}
        by_seed = {r['seed']: r for r in res['realizations']}
        res['fd_vs_exact_reference_max_abs_pct'] = {
            p: max(abs(v['ref_fd_err_pct'][p] - by_seed[s]['err_pct'][p]) for s, v in pinn.items() if s in by_seed)
            for p in PARAMS}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'A1_reference_mc.json').write_text(json.dumps(res, indent=1, default=float))
    plot(res, pinn, out / 'A1_reference_mc.png')

    sm = res['summary']
    print(f"{len(args.seeds)} realizations, held-out A = {res['design']['A_heldout']}")
    for p in PARAMS:
        s = sm['params_pct'][p]['log']
        print(f"  {p:<4} bias {s['bias']:+5.2f} ± {s['bias_se']:.2f}%  SD {s['sd']:5.2f}%  CRB "
              f"{sm['params_pct'][p]['rms_crb_pct']:5.2f}%  Laplace {s['rms_predicted_sd']:5.2f}%  cover "
              f"{s['covered']}/{s['n']} [{s['coverage_ci95'][0]:.2f}, {s['coverage_ci95'][1]:.2f}]")
    s = sm['GTE_pct']
    print(f"  GTE  bias {s['bias']:+5.2f} ± {s['bias_se']:.2f}%  SD {s['sd']:5.2f}%  CRB {s['rms_crb_pct']:5.2f}%  "
          f"Laplace {s['rms_predicted_sd']:5.2f}%  cover {s['covered']}/{s['n']} "
          f"[{s['coverage_ci95'][0]:.2f}, {s['coverage_ci95'][1]:.2f}]")
    for q in ('ATE_L2_pct', 'tau_L2_pct'):
        s = sm[q]
        print(f"  {q:<10} RMS error {s['rms']:5.2f}%  bound {s['rms_crb']:5.2f}%  "
              f"pointwise coverage {s['mean_pointwise_coverage']:.3f}")
    print(f"  fits at a bound: {sm['n_at_bound']}")
    if pinn:
        print('  FD-staged vs exact reference, max |diff| (pp):', res['fd_vs_exact_reference_max_abs_pct'])
    print(f'  -> {out}')


if __name__ == '__main__':
    main()
