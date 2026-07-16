#!/usr/bin/env python3
"""inset_perwindow.py -- in-set-only corrected per-window evaluation (Supp F addition).

Mirrors stage_tab8 of run_v2_honest.py (identical fresh-nominal split: 30 calibration /
30 test negatives, 95% calibration-quantile threshold) with attack positives filtered to
episodes satisfying I>=rho, D_leg<=eps, E<=E_max. Methods: Q-Direct, RBF, PCA-norm.
Families with <2 in-set episodes on a seed are excluded from that seed's mean (N_s column).

Run from the repository root:  python3 asqdt/scripts/inset_perwindow.py
Output: asqdt/results/v2_inset_perwindow.json + LaTeX rows on stdout.
"""
import pickle, sys, os, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
from asqdt import config as cfg
from asqdt.evaluation.fastq import gram_from_sv
from asqdt.scripts.run_v2_honest import fresh_noms, kscore, ensure_tank_stats
from asqdt.scripts.run_v2_revision import load_or_build_mp, RESULTS

ALPHA = 0.05
LABEL = {'fdi':'FDI','command':'Sinusoidal','ramp':'Ramp','composite':'Composite',
         'cstr_fdi':'FDI','cstr_thermal':'Thermal','cstr_concentration':'Conc.',
         'tank_bias':'Bias','tank_oscillation':'Oscill.','tank_drift':'Drift'}

def main():
    MP = load_or_build_mp(set()); ensure_tank_stats(MP)
    out = {}
    for bname in ('CNC','CSTR','Tank'):
        P = cfg.ATTACK_SET[bname]; rho,eps,emax = P['rho'],P['epsilon'],P['E_max']
        fams = list(MP[bname][cfg.SEEDS[0]]['data']['atk'].keys())
        acc = {a:{m:{'f1':[],'auc':[]} for m in ('q','rbf','pcanorm')} for a in fams}
        ncount = {a: [] for a in fams}
        for seed in cfg.SEEDS:
            mp = MP[bname][seed]; d = mp['data']
            F = (fresh_noms(bname, seed, d, 60) - mp['mu']) @ mp['V']
            Xcal, Xtest = F[:30], F[30:]
            Xn_p, gamma = mp['Xn_p'], mp['gamma']
            Kr = sk_rbf(Xn_p, Xn_p, gamma)
            fns = {'q':   lambda X: kscore(mp['K_q'], gram_from_sv(mp['S_nom'], mp['qcache'].sv(X))),
                   'rbf': lambda X: kscore(Kr, sk_rbf(Xn_p, X, gamma)),
                   'pcanorm': lambda X: np.linalg.norm(X, axis=1)}
            H = {m:(fn, np.percentile(fn(Xcal),100*(1-ALPHA)), fn(Xtest)) for m,fn in fns.items()}
            for a in fams:
                ai = d['atk'][a]
                ok = np.array([(ai['impacts'][e]>=rho) and (ai['legacy_responses'][e]<=eps)
                               and (ai['energies'][e]<=emax) for e in range(len(ai['impacts']))])
                ncount[a].append(int(ok.sum()))
                if ok.sum() < 2: continue
                Xa = (np.nan_to_num(np.array(ai['windows'])[ok]) - mp['mu']) @ mp['V']
                y = np.concatenate([np.zeros(30), np.ones(len(Xa))])
                for m,(fn,tau,sneg) in H.items():
                    s = np.concatenate([sneg, fn(Xa)])
                    acc[a][m]['f1'].append(f1_score(y,(s>tau).astype(int),zero_division=0))
                    acc[a][m]['auc'].append(roc_auc_score(y,s))
        out[bname] = {}
        for a in fams:
            ns = len(acc[a]['q']['auc'])
            row = {'nbar': float(np.mean(ncount[a])), 'seeds': ns}
            for m in ('q','rbf','pcanorm'):
                v = acc[a][m]
                if v['auc']:
                    row[m] = {k:(float(np.mean(v[k])), float(np.std(v[k]))) for k in ('f1','auc')}
            out[bname][a] = row
            if ns:
                star = r'$^{\ast}$' if ns < 5 else ''
                print(f"{bname} & {LABEL[a]}{star} & {row['nbar']:.1f} & {ns} & "
                      f"{row['q']['auc'][0]:.2f}$\\pm${row['q']['auc'][1]:.2f} & {row['q']['f1'][0]:.2f} & "
                      f"{row['rbf']['auc'][0]:.2f}$\\pm${row['rbf']['auc'][1]:.2f} & {row['rbf']['f1'][0]:.2f} & "
                      f"{row['pcanorm']['auc'][0]:.2f} & {row['pcanorm']['f1'][0]:.2f} \\\\")
    with open(os.path.join(RESULTS, 'v2_inset_perwindow.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print("saved", os.path.join(RESULTS, 'v2_inset_perwindow.json'))

if __name__ == '__main__':
    main()
