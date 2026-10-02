"""Mechanistic counterfactual study, do(A = a), on the linear model (dopamine_pinn.causal).

    python scripts/run_causal_linear.py --quick              # smoke test, ~2 min CPU
    python scripts/run_causal_linear.py                      # paper settings (GPU advised)
    python scripts/run_causal_linear.py --obs-seeds 4001 4002 4003

Writes causal_out/<obs_seed>/results.json and figures; a finished seed is skipped.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dopamine_pinn import config as cfg  # noqa: E402
from dopamine_pinn import causal as ca  # noqa: E402
from dopamine_pinn.pinn import TrainConfig  # noqa: E402


def public(d):
    if isinstance(d, dict):
        return {k: public(v) for k, v in d.items() if not str(k).startswith('_')}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='tiny schedule (not results)')
    ap.add_argument('--obs-seeds', type=int, nargs='+', default=[ca.OBS_SEED])
    ap.add_argument('--net-seed', type=int, default=cfg.SEED)
    ap.add_argument('--no-reference', action='store_true')
    ap.add_argument('--out', default='causal_out')
    args = ap.parse_args()

    tc = TrainConfig.quick(verbose=True, log_every=2) if args.quick else TrainConfig()
    out = Path(args.out) / ('quick' if args.quick else '')
    for s in args.obs_seeds:
        d = out / f'obs{s}'
        if (d / 'results.json').exists():
            print(f'obs seed {s}: done, skipping')
            continue
        print(f'=== obs seed {s} ===')
        res, priv = ca.run_study(tc, obs_seed=s, seed=args.net_seed, with_reference=not args.no_reference)
        d.mkdir(parents=True, exist_ok=True)
        ca.plot_study(res, priv, d)
        (d / 'results.json').write_text(json.dumps(public(res), indent=1, default=float))
        print(f'  -> {d}  ({res["minutes"]:.1f} min)')


if __name__ == '__main__':
    main()
