#!/usr/bin/env python3
"""
External validation runner for public ICS datasets (SWaT / WADI / HAI).

Usage:
  python -m asqdt.scripts.run_swat_validation NORMAL.csv ATTACK.csv \
      [--ds 5] [--wl 10] [--seeds 42 123 256 789 1024] [--nrows N]

For SWaT.A1&A2 (Dec 2015): NORMAL = SWaT_Dataset_Normal_v1.csv,
ATTACK = SWaT_Dataset_Attack_v0.csv (convert xlsx to csv first, or pass xlsx).
Default ds=5 turns 1-second sampling into 5-second medians.

Prints pooled AUC/F1/FPR per method (LaTeX row) + concentration diagnostic,
and saves asqdt/results/swat_validation.json with per-scenario hits.
"""
import sys, os, json, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from asqdt.external.icsdata import run_protocol
from asqdt.scripts.run_v2_revision import RESULTS

METHODS = ['Q-Direct', 'RBF', 'Laplacian', 'OC-SVM^t', 'IForest^t', 'AE', 'OC-SVM', 'Res-norm']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('normal'); ap.add_argument('attack')
    ap.add_argument('--ds', type=int, default=5)
    ap.add_argument('--wl', type=int, default=10)
    ap.add_argument('--nrows', type=int, default=None)
    ap.add_argument('--qscale', type=float, default=1.0,
                    help='bandwidth-controlled input rescale for the quantum path only')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 256, 789, 1024])
    a = ap.parse_args()

    agg = {m: {'auc': [], 'f1': [], 'fpr': []} for m in METHODS}
    out = {'runs': []}
    for seed in a.seeds:
        print(f"\n== seed {seed} ==")
        res = run_protocol(a.normal, a.attack, ds=a.ds, W_L=a.wl,
                           seed=seed, nrows=a.nrows, q_scale=a.qscale)
        out['runs'].append(res)
        for m in METHODS:
            for k in ('auc', 'f1', 'fpr'):
                agg[m][k].append(res['results'][m][k])
        c = res['concentration']
        print(f"  concentration: k(ref,ref)={c['ref_ref'][0]:.3f}±{c['ref_ref'][1]:.3f}"
              f"  k(ref,nom)={c['ref_nom'][0]:.3f}±{c['ref_nom'][1]:.3f}"
              f"  k(ref,atk)={c['ref_atk'][0]:.3f}±{c['ref_atk'][1]:.3f}")
        for m in METHODS:
            r = res['results'][m]
            print(f"  {m:10s} AUC={r['auc']:.2f} F1={r['f1']:.2f} FPR={r['fpr']:.2f} TPR={r['tpr']:.2f}")

    print(f"\n== {len(a.seeds)}-seed aggregate (LaTeX row: method & AUC & F1 & FPR) ==")
    for m in METHODS:
        v = agg[m]
        print(f"  {m} & {np.mean(v['auc']):.2f}$\\pm${np.std(v['auc']):.2f}"
              f" & {np.mean(v['f1']):.2f}$\\pm${np.std(v['f1']):.2f}"
              f" & {np.mean(v['fpr']):.2f}$\\pm${np.std(v['fpr']):.2f} \\\\")
    path = os.path.join(RESULTS, 'swat_validation.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {path}")

if __name__ == '__main__':
    main()
