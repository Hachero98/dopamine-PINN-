"""Tables over many studies: PINN against the reference on the same records, and both
against the Cramer-Rao bound.

    rows = load_runs('runs', 'linear', 'fast')
    summary = compare(rows)          # paired differences, RMSE ratios, coverage
    print_summary(summary)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import beta


def load_runs(out, mech_name, profile_name, net_seed=None):
    """Results of all finished studies of one mechanism and profile (sorted by seed)."""
    rows = []
    for p in sorted(Path(out, mech_name, profile_name).glob('obs*_net*/results.json')):
        r = json.loads(p.read_text())
        if net_seed is None or r['seeds']['net'] == net_seed:
            r['_dir'] = str(p.parent)
            rows.append(r)
    return rows


def clopper_pearson(k, n, level=0.95):
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def _paired(p, r):
    p, r = np.asarray(p), np.asarray(r)
    d = p - r
    n = d.size
    out = {'n': int(n), 'rmse_pinn': float(np.sqrt(np.mean(p**2))), 'rmse_ref': float(np.sqrt(np.mean(r**2))),
           'mean_diff': float(d.mean())}
    out['ratio'] = out['rmse_pinn'] / out['rmse_ref'] if out['rmse_ref'] > 0 else np.nan
    if n >= 3:
        se = d.std(ddof=1) / np.sqrt(n)
        tq = stats.t.ppf(0.975, n - 1)
        out['mean_diff_ci95'] = [float(d.mean() - tq * se), float(d.mean() + tq * se)]
        out['wilcoxon_p'] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        idx = np.random.default_rng(0).integers(0, n, (10_000, n))
        boot = np.sqrt(np.mean(p[idx]**2, 1) / np.maximum(np.mean(r[idx]**2, 1), 1e-300))
        out['ratio_ci95'] = np.percentile(boot, [2.5, 97.5]).tolist()
    return out


def compare(rows):
    """Paired PINN vs reference over studies; errors in %, log scale for parameters."""
    if not rows:
        return {}
    params = rows[0]['mechanism']['params']
    col = lambda f: np.array([f(r) for r in rows], float)
    out = {'mechanism': rows[0]['mechanism']['name'], 'profile': rows[0]['profile']['name'],
           'obs_seeds': [r['seeds']['obs'] for r in rows], 'quantities': {}}
    for q in params:
        e = out['quantities'][q] = _paired(col(lambda r: r['pinn']['stage4']['err_log_pct'][q]),
                                           col(lambda r: r['reference']['stage4']['err_log_pct'][q]))
        e['bound_rms'] = float(np.sqrt(np.mean(col(lambda r: r['bound']['sd_log_pct'][q]) ** 2)))
    for name, key in (('GTE (PDE at PINN params)', 'pinn_pde'), ('GTE (network)', 'network')):
        e = out['quantities'][name] = _paired(col(lambda r: r['heldout'][key]['GTE_err_pct']),
                                              col(lambda r: r['heldout']['reference']['GTE_err_pct']))
        e['bound_rms'] = float(np.sqrt(np.mean(col(lambda r: r['bound']['GTE_sd_pct']) ** 2)))
    for name, key in (('tau L2 (PDE at PINN params)', 'pinn_pde'), ('tau L2 (network)', 'network')):
        p, r_ = col(lambda r: r['heldout'][key]['tau_L2_pct']), col(lambda r: r['heldout']['reference']['tau_L2_pct'])
        out['quantities'][name] = {'n': len(rows), 'rms_pinn': float(np.sqrt(np.mean(p**2))),
                                   'rms_ref': float(np.sqrt(np.mean(r_**2))),
                                   'ratio': float(np.sqrt(np.mean(p**2) / np.mean(r_**2))),
                                   'bound_rms': float(np.sqrt(np.mean(col(lambda r: r['bound']['tau_L2_rmse_pct'])**2)))}
    c = int(sum(r['heldout']['reference']['GTE_covered'] for r in rows))
    out['reference_GTE_coverage'] = {'covered': c, 'n': len(rows), 'ci95': clopper_pearson(c, len(rows))}
    dg = [r['diagnostics'] for r in rows]
    out['diagnostics_median'] = {k: float(np.median([d[k] for d in dg])) for k in dg[0]}
    out['minutes_median'] = float(np.median(col(lambda r: r['minutes'])))
    return out


def print_summary(s):
    if not s:
        print('no finished studies')
        return
    print(f"{s['mechanism']} | {s['profile']} | {len(s['obs_seeds'])} realizations | "
          f"{s['minutes_median']:.1f} min per study")
    print(f"  {'quantity':<28} {'PINN':>7} {'ref':>7} {'bound':>7} {'ratio':>6}  ratio 95% CI")
    for q, e in s['quantities'].items():
        pin, ref = e.get('rmse_pinn', e.get('rms_pinn')), e.get('rmse_ref', e.get('rms_ref'))
        ci = e.get('ratio_ci95')
        print(f"  {q:<28} {pin:7.2f} {ref:7.2f} {e['bound_rms']:7.2f} {e['ratio']:6.2f}  "
              + (f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ''))
    cv = s['reference_GTE_coverage']
    print(f"  reference GTE coverage {cv['covered']}/{cv['n']} [{cv['ci95'][0]:.2f}, {cv['ci95'][1]:.2f}]")
    d = s['diagnostics_median']
    print(f"  PINN data misfit / sigma^2 {d['data_misfit_pinn_over_sigma2']:.3f} (reference "
          f"{d['data_misfit_ref_over_sigma2']:.3f}); residual / uptake {d['residual_over_uptake']:.3f}; "
          f"weighted data / residual {d['weighted_data_over_residual']:.0f}")
