#!/usr/bin/env python3
"""exhaustive_masks.py -- exhaustive budgeted sensor-mask evaluation (Supp P addition).

Evaluates ALL C(6,4)=15 CNC masks and all C(3,2)=3 CSTR masks per seed under kernels
frozen at the full-sensor design (theta, gamma from the canonical pipeline), PCA re-fit
per mask -- the protocol of stage_C in run_v2_honest.py -- and reports the exact rank of
the greedy and top-variance masks (1 = largest batch MMD proxy).

Run from the repository root:  python3 asqdt/scripts/exhaustive_masks.py
Output: asqdt/results/v2_exhaustive_masks.json + rank table on stdout.
"""
import sys, os, json, warnings
from itertools import combinations
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from asqdt import config as cfg
from asqdt.scripts.run_v2_honest import eval_mask2, ensure_tank_stats
from asqdt.scripts.run_v2_revision import load_or_build_mp, RESULTS
from asqdt.codesign.sensors_v2 import mask_from

def main():
    MP = load_or_build_mp(set()); ensure_tank_stats(MP)
    sel = {'CNC': json.load(open(os.path.join(RESULTS,'v2_honest_C_CNC_b4.json')))['selected'],
           'CSTR': json.load(open(os.path.join(RESULTS,'v2_honest_C_CSTR_b2.json')))['selected']}
    out = {}
    for bname, p, budget in (('CNC',6,4), ('CSTR',3,2)):
        masks = list(combinations(range(p), budget))
        Q = np.zeros((len(cfg.SEEDS), len(masks))); R = np.zeros_like(Q)
        for i, seed in enumerate(cfg.SEEDS):
            mp = MP[bname][seed]
            for j, m in enumerate(masks):
                Q[i,j], R[i,j] = eval_mask2(bname, seed, mask_from(list(m), p), mp['theta'], mp['gamma'])
        print(f"\n== {bname}: exhaustive {len(masks)} masks x {len(cfg.SEEDS)} seeds ==")
        ranks = {'greedy': {'q': [], 'r': []}, 'topvar': {'q': [], 'r': []}}
        for i, seed in enumerate(cfg.SEEDS):
            for cname in ('greedy','topvar'):
                j = masks.index(tuple(sel[bname][str(seed)][cname]))
                rq = int(1 + np.sum(Q[i] > Q[i,j])); rr = int(1 + np.sum(R[i] > R[i,j]))
                ranks[cname]['q'].append(rq); ranks[cname]['r'].append(rr)
            print(f"  seed {seed}: greedy Q-rank {ranks['greedy']['q'][-1]:2d}/{len(masks)} "
                  f"RBF-rank {ranks['greedy']['r'][-1]:2d} | topvar Q-rank {ranks['topvar']['q'][-1]:2d} "
                  f"RBF-rank {ranks['topvar']['r'][-1]:2d}")
        Qm, Rm = Q.mean(0), R.mean(0)
        print(f"  seed-mean Q span over masks: [{Qm.min():+.4f},{Qm.max():+.4f}] "
              f"(typ. cross-seed std {Q.std(0).mean():.4f}); "
              f"RBF span: [{Rm.min():+.4f},{Rm.max():+.4f}] (typ. std {R.std(0).mean():.4f})")
        out[bname] = {'masks': [list(m) for m in masks], 'Q': Q.tolist(), 'R': R.tolist(),
                      'ranks': ranks}
    with open(os.path.join(RESULTS, 'v2_exhaustive_masks.json'), 'w') as f:
        json.dump(out, f)
    print("\nsaved", os.path.join(RESULTS, 'v2_exhaustive_masks.json'))

if __name__ == '__main__':
    main()
