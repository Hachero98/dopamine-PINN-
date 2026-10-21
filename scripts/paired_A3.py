"""Paper 2, tasks A3 and A7: PINN against the reference estimator on the same records.

Reads the Colab results of dopamine_PINN_causal.ipynb (obs<seed>_net<seed>/results.json).
For every realization with network seed 1234 it pairs the PINN (stage 4 parameters;
stage 5 effect at the held-out dose, both as the network and as the PDE solved at the
PINN parameters) with the FD reference estimator fitted to the same observations, and
reports
  * paired differences PINN - reference: mean with 95% t interval, Wilcoxon signed-rank p;
  * RMSE of each and their ratio (bootstrap 95% interval), against the Cramer-Rao bound;
  * coverage of the reference's 95% Laplace interval and of the same-width interval
    centred on the PINN (the PINN has no interval of its own; A7);
  * network-seed spread (seeds 1234-1238 on realization 4001) against realization spread.

    python scripts/paired_A3.py dopamine_PINN_causal-2
Writes results/paper2_linear/A3_paired.json and A3_paired.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dopamine_pinn.counterfactual import clopper_pearson  # noqa: E402

PARAMS = ('D', 'k', 'eta')
Z = 1.96


def load(pinn_dir):
    real, seeds4001 = {}, {}
    for p in sorted(Path(pinn_dir).glob('obs*_net*/results.json')):
        r = json.loads(p.read_text())
        s5, ref = r['stage5'], r['reference']['stage4']
        gte0 = s5['truth']['GTE']
        row = {
            'obs_seed': r['design']['obs_seed'], 'net_seed': r['design']['net_seed'],
            # parameters: errors of the log-parameter in % (comparable with the Laplace SD)
            **{f'pinn_{q}': 100 * np.log1p(r['pinn']['stage4']['err_pct'][q] / 100) for q in PARAMS},
            **{f'ref_{q}': 100 * np.log1p(ref['err_pct'][q] / 100) for q in PARAMS},
            **{f'refsd_{q}': 100 * ref['sd_log'][q] for q in PARAMS},
            'pinn_GTE_pde': s5['pinn_calibrated_pde']['GTE_err_pct'],
            'pinn_GTE_net': s5['pinn_network']['GTE_err_pct'],
            'ref_GTE': s5['reference_calibrated_pde']['GTE_err_pct'],
            'refsd_GTE': 100 * s5['reference_calibrated_pde']['GTE_sd'] / gte0,
            'pinn_tau_pde': s5['pinn_calibrated_pde']['E_L2_tau_pct'],
            'pinn_tau_net': s5['pinn_network']['E_L2_tau_pct'],
            'ref_tau': s5['reference_calibrated_pde']['E_L2_tau_pct'],
            'pinn_SA_net': s5['pinn_network']['E_L2_S_A_pct'],
        }
        if row['net_seed'] == 1234:
            real[row['obs_seed']] = row
        if row['obs_seed'] == 4001:
            seeds4001[row['net_seed']] = row
    return real, seeds4001


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = d.size
    se = d.std(ddof=1) / np.sqrt(n)
    tq = stats.t.ppf(0.975, n - 1)
    return {'mean': float(d.mean()), 'ci95': [float(d.mean() - tq * se), float(d.mean() + tq * se)],
            'sd': float(d.std(ddof=1)), 'wilcoxon_p': float(stats.wilcoxon(d).pvalue),
            'corr_pinn_ref': float(np.corrcoef(a, b)[0, 1])}


def rmse_ratio(a, b, n_boot=20_000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    rmse = lambda x: np.sqrt(np.mean(x**2))
    idx = np.random.default_rng(seed).integers(0, a.size, (n_boot, a.size))
    boot = np.sqrt(np.mean(a[idx] ** 2, 1) / np.mean(b[idx] ** 2, 1))
    return {'rmse_pinn': float(rmse(a)), 'rmse_ref': float(rmse(b)), 'ratio': float(rmse(a) / rmse(b)),
            'ratio_ci95': np.percentile(boot, [2.5, 97.5]).tolist()}


def coverage(err, sd):
    c = int(np.sum(np.abs(err) <= Z * np.asarray(sd)))
    return {'covered': c, 'n': len(err), 'coverage': c / len(err), 'ci95': clopper_pearson(c, len(err))}


def analyse(real, seeds4001, crb=None):
    R = [real[s] for s in sorted(real)]
    col = lambda k: np.array([r[k] for r in R])
    out = {'n_realizations': len(R), 'obs_seeds': sorted(real), 'quantities': {}}
    for q, pk, rk, sk in [('D', 'pinn_D', 'ref_D', 'refsd_D'), ('k', 'pinn_k', 'ref_k', 'refsd_k'),
                          ('eta', 'pinn_eta', 'ref_eta', 'refsd_eta'),
                          ('GTE (PDE at PINN params)', 'pinn_GTE_pde', 'ref_GTE', 'refsd_GTE'),
                          ('GTE (network)', 'pinn_GTE_net', 'ref_GTE', 'refsd_GTE')]:
        out['quantities'][q] = {'paired_diff': paired(col(pk), col(rk)), **rmse_ratio(col(pk), col(rk)),
                                'bias_pinn': float(col(pk).mean()), 'bias_ref': float(col(rk).mean()),
                                'coverage_ref': coverage(col(rk), col(sk)),
                                'coverage_pinn_ref_width': coverage(col(pk), col(sk))}
    for q, pk in [('tau L2 (PDE at PINN params)', 'pinn_tau_pde'), ('tau L2 (network)', 'pinn_tau_net')]:
        out['quantities'][q] = {'paired_diff': paired(col(pk), col('ref_tau')),
                                'rms_pinn': float(np.sqrt(np.mean(col(pk) ** 2))),
                                'rms_ref': float(np.sqrt(np.mean(col('ref_tau') ** 2))),
                                'ratio': float(np.sqrt(np.mean(col(pk) ** 2) / np.mean(col('ref_tau') ** 2)))}
    S = [seeds4001[n] for n in sorted(seeds4001)]
    out['network_vs_realization_spread'] = {
        q: {'sd_over_net_seeds_4001': float(np.std([r[k] for r in S], ddof=1)),
            'sd_over_realizations_pinn': float(col(k).std(ddof=1)),
            'sd_over_realizations_ref': float(col(rk).std(ddof=1))}
        for q, k, rk in [('k', 'pinn_k', 'ref_k'), ('eta', 'pinn_eta', 'ref_eta'), ('D', 'pinn_D', 'ref_D'),
                         ('GTE_pde', 'pinn_GTE_pde', 'ref_GTE'), ('tau_pde', 'pinn_tau_pde', 'ref_tau')]}
    out['network_vs_realization_spread']['n_net_seeds'] = len(S)
    worst = sorted(R, key=lambda r: -abs(r['pinn_GTE_pde'] - r['ref_GTE']))[:5]
    out['largest_GTE_differences'] = [{k: r[k] for k in ('obs_seed', 'pinn_k', 'ref_k', 'pinn_eta', 'ref_eta',
                                                         'pinn_GTE_pde', 'ref_GTE', 'refsd_GTE')} for r in worst]
    if crb:
        out['crb_pct'] = crb
    out['realizations'] = R
    out['net_seeds_4001'] = S
    return out


def plot(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = res['realizations']
    col = lambda k: np.array([r[k] for r in R])
    crb = res.get('crb_pct', {})
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    for a, (q, pk, rk) in zip(ax[:3], [('k', 'pinn_k', 'ref_k'), ('eta', 'pinn_eta', 'ref_eta'),
                                       ('GTE', 'pinn_GTE_pde', 'ref_GTE')]):
        x, y = col(rk), col(pk)
        lim = 1.1 * max(np.abs(x).max(), np.abs(y).max())
        a.plot([-lim, lim], [-lim, lim], 'k-', lw=0.6)
        if q in crb:
            c = Z * crb[q]
            a.axvspan(-c, c, color='0.92', zorder=0, label='±1.96 CRB')
            a.axhspan(-c, c, color='0.92', zorder=0)
        a.scatter(x, y, s=18, c='C3')
        for r in R:
            if abs(r[pk] - r[rk]) > 6:
                a.annotate(str(r['obs_seed']), (r[rk], r[pk]), fontsize=7, xytext=(3, 3), textcoords='offset points')
        d = res['quantities'][{'GTE': 'GTE (PDE at PINN params)'}.get(q, q)]['paired_diff']
        a.set_title(f'{q}: PINN − ref {d["mean"]:+.1f} [{d["ci95"][0]:+.1f}, {d["ci95"][1]:+.1f}] pp, r = '
                    f'{d["corr_pinn_ref"]:.2f}', fontsize=9)
        a.set_xlabel(f'reference error (%)')
        a.set_ylabel(f'PINN error (%)')
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.legend(fontsize=8, loc='upper left')
    a = ax[3]
    x = np.arange(len(R))
    a.bar(x - 0.2, col('ref_tau'), 0.4, label='reference', color='C0')
    a.bar(x + 0.2, col('pinn_tau_pde'), 0.4, label='PINN (PDE at λ̂)', color='C3')
    if 'tau_L2' in crb:
        a.axhline(crb['tau_L2'], color='k', ls='--', lw=0.8, label='RMS bound')
    a.set_xticks(x)
    a.set_xticklabels([str(r['obs_seed'])[-2:] for r in R], fontsize=7)
    a.set_xlabel('realization (40xx)')
    a.set_ylabel('L2 error of τ at A = 0.5 (%)')
    a.set_title('Held-out effect field, paired')
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    pinn_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'dopamine_PINN_causal')
    out = Path(sys.argv[2] if len(sys.argv) > 2 else 'results/paper2_linear')
    real, seeds4001 = load(pinn_dir)
    crb = None
    a1 = out / 'A1_reference_mc.json'
    if a1.exists():
        sm = json.loads(a1.read_text())['summary']
        crb = {q: sm['params_pct'][q]['rms_crb_pct'] for q in PARAMS}
        crb.update({'GTE': sm['GTE_pct']['rms_crb_pct'], 'tau_L2': sm['tau_L2_pct']['rms_crb']})
    res = analyse(real, seeds4001, crb)
    res['source'] = str(pinn_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'A3_paired.json').write_text(json.dumps(res, indent=1, default=float))
    plot(res, out / 'A3_paired.png')

    print(f"{res['n_realizations']} realizations, PINN - reference on the same records (pp; log scale for parameters)")
    for q, v in res['quantities'].items():
        d = v['paired_diff']
        line = f"  {q:<28} diff {d['mean']:+6.2f} [{d['ci95'][0]:+6.2f}, {d['ci95'][1]:+6.2f}]  p {d['wilcoxon_p']:.3f}  r {d['corr_pinn_ref']:.2f}"
        if 'rmse_pinn' in v:
            line += (f"  RMSE {v['rmse_pinn']:5.2f} vs {v['rmse_ref']:5.2f} (x{v['ratio']:.2f} "
                     f"[{v['ratio_ci95'][0]:.2f}, {v['ratio_ci95'][1]:.2f}])  cover ref {v['coverage_ref']['covered']}/"
                     f"{v['coverage_ref']['n']} PINN {v['coverage_pinn_ref_width']['covered']}/{v['coverage_pinn_ref_width']['n']}")
        else:
            line += f"  RMS {v['rms_pinn']:5.2f} vs {v['rms_ref']:5.2f} (x{v['ratio']:.2f})"
        print(line)
    print('  spread: SD over network seeds (4001) | over realizations PINN | ref')
    for q, v in res['network_vs_realization_spread'].items():
        if isinstance(v, dict):
            print(f"    {q:<8} {v['sd_over_net_seeds_4001']:5.2f} | {v['sd_over_realizations_pinn']:5.2f} | "
                  f"{v['sd_over_realizations_ref']:5.2f}")
    print('  largest GTE differences:', [(r['obs_seed'], round(r['pinn_GTE_pde'], 1), round(r['ref_GTE'], 1))
                                         for r in res['largest_GTE_differences']])


if __name__ == '__main__':
    main()
