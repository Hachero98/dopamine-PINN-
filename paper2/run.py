"""Run counterfactual studies and summarise them.

    python paper2/run.py --model linear --profile fast --seeds 4001 4002 4003
    python paper2/run.py --model mm_capacity --profile fast --seeds 4001-4005
    python paper2/run.py --model linear --profile fast --w-res 30000 --tag wres3e4 --seeds 4009 4010
    python paper2/run.py --model linear --profile fast --summary          # table + figure only

Models: linear, mm_capacity, mm_competitive. Profiles: smoke, fast, paper.
Results go to <out>/<model>/<profile>[_<tag>]/obs<seed>_net<seed>/; finished studies are
skipped, so an interrupted run resumes where it stopped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cf  # noqa: E402
from cf.analysis import load_runs, compare, print_summary  # noqa: E402
from cf.plots import plot_study, plot_compare  # noqa: E402


def seeds(spec):
    out = []
    for s in spec:
        if '-' in s:
            a, b = map(int, s.split('-'))
            out += list(range(a, b + 1))
        else:
            out.append(int(s))
    assert max(out) < 4101, 'realization seeds must stay below 4101 (dose j uses seed + 100 j)'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='linear', choices=sorted(cf.MECHANISMS))
    ap.add_argument('--profile', default='fast', choices=sorted(cf.PROFILES))
    ap.add_argument('--seeds', nargs='+', default=['4001'], help='e.g. 4001 4002 or 4001-4020')
    ap.add_argument('--net-seed', type=int, default=1234)
    ap.add_argument('--w-res', type=float, help='residual weight (default: the profile\'s)')
    ap.add_argument('--w-data', type=float, help='data weight (default: the profile\'s)')
    ap.add_argument('--tag', default='', help='suffix for the profile folder, e.g. wres3e4')
    ap.add_argument('--out', default='paper2/runs')
    ap.add_argument('--summary', action='store_true', help='only summarise finished studies')
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()

    prof = cf.PROFILES[args.profile]
    changes = {k: v for k, v in (('w_res', args.w_res), ('w_data', args.w_data)) if v is not None}
    if changes or args.tag:
        prof = prof.replace(name=f'{prof.name}_{args.tag}' if args.tag else prof.name, **changes)
        assert args.tag or not changes, 'give --tag when changing weights, so results do not mix'

    if not args.summary:
        for s in seeds(args.seeds):
            cf.run_study(args.model, prof, obs_seed=s, net_seed=args.net_seed, out=args.out)
            d = cf.study.study_dir(args.out, args.model, prof.name, s, args.net_seed)
            if not args.no_figures and not (d / 'effect_fields.png').exists():
                plot_study(d)

    rows = load_runs(args.out, args.model, prof.name, args.net_seed)
    s = compare(rows)
    print_summary(s)
    if rows and not args.no_figures:
        plot_compare(rows, Path(args.out) / args.model / prof.name / 'compare.png')


if __name__ == '__main__':
    main()
