#!/usr/bin/env python3
"""
run_test_split_allmethods.py — extends the disjoint attack-split evaluation
(run_test_split.py) to the FULL detector suite of the corrected protocol:
Q-Direct, RBF, Laplacian, tuned OC-SVM, tuned IForest, AE, default OC-SVM,
and Res-norm, all with design choices frozen (same mp cache, same tuned
hyperparameters as the main tables) and scored on the 20 fresh test attack
positives (episodes 20..39) against 30 held-out nominal negatives.

Output: asqdt/results/v2_test_split_allmethods.json
Run:    python3 asqdt/scripts/run_test_split_allmethods.py
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.metrics import f1_score, roc_auc_score

from asqdt import config as cfg
from asqdt.scripts.run_v2_revision import load_or_build_mp, RESULTS
from asqdt.scripts.run_v2_honest import fresh_noms, ensure_tank_stats, honest_methods
from asqdt.scripts.run_test_split import gen_attack_windows, LAB

METHODS = ['q', 'rbf', 'lap', 'ocsvm_t', 'iforest_t', 'ae', 'ocsvm_d', 'resnorm']


def main():
    MP = load_or_build_mp(set())
    ensure_tank_stats(MP)
    out = {}
    for bname in ('CNC', 'CSTR', 'Tank'):
        fams = list(MP[bname][cfg.SEEDS[0]]['data']['atk'].keys())
        acc = {a: {m: dict(auc=[], f1=[]) for m in METHODS} for a in fams}
        for seed in cfg.SEEDS:
            mp = MP[bname][seed]
            mu, V = mp['mu'], mp['V']
            F = (fresh_noms(bname, seed, mp['data'], 60) - mu) @ V
            Xcal, Xneg = F[:30], F[30:]
            meth = honest_methods(mp, Xcal, Xneg)   # frozen models, calibrated taus
            test_fams = gen_attack_windows(bname, seed, mp, 20, 40)
            for a in fams:
                Xa = (test_fams[a]['W'] - mu) @ V
                y = np.concatenate([np.zeros(len(Xneg)), np.ones(len(Xa))])
                for m in METHODS:
                    fn, tau, s_neg = meth[m]
                    s_pos = fn(Xa)
                    sc = np.concatenate([s_neg, s_pos])
                    acc[a][m]['auc'].append(float(roc_auc_score(y, sc)))
                    acc[a][m]['f1'].append(float(f1_score(y, (sc > tau).astype(int),
                                                          zero_division=0)))
            print(f"  {bname} seed={seed} done")
        out[bname] = {a: {m: dict(auc=[float(np.mean(acc[a][m]['auc'])),
                                       float(np.std(acc[a][m]['auc']))],
                                  f1=[float(np.mean(acc[a][m]['f1'])),
                                      float(np.std(acc[a][m]['f1']))])
                          for m in METHODS} for a in fams}

    with open(os.path.join(RESULTS, 'v2_test_split_allmethods.json'), 'w') as f:
        json.dump(out, f, indent=1)

    print("\n=== TEST-SPLIT (fresh positives ep20..39) AUC per method ===")
    hdr = f"{'bench/family':22s}" + "".join(f"{m:>10s}" for m in METHODS)
    print(hdr)
    for bname in out:
        for a in out[bname]:
            row = f"{bname+'/'+LAB[a]:22s}"
            for m in METHODS:
                row += f"{out[bname][a][m]['auc'][0]:10.2f}"
            print(row)


if __name__ == '__main__':
    main()
