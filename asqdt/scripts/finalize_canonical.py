#!/usr/bin/env python3
"""
finalize_canonical.py — prints every remaining number for the revised paper
that must come from the CANONICAL code state (the authors' local copy that
reproduces the published Table IV). Run AFTER run_v2_revision (uses its
mp_cache.pkl). Runtime ~2 min. Items:

  A. Per-seed Q vs RBF batch MMD per family + exact paired sign tests
     -> verifies/fills the significance sentence in Finding 1.
  B. Fidelity-kernel concentration diagnostic (CNC seed 42) + mean off-
     diagonal reference similarity k-bar + self-inclusion bias + fresh-
     nominal FPR at the initial (invalid) thresholds
     -> fills [K1] [K2] [K3] and [C1] [C2] [C3] in main_v2.tex and Supp L.
  C. Encoding-scale probe (rescale kernel inputs by c)
     -> fills [P1]-[P4] in main_v2.tex.
  D. In-set-only batch MMD (Q and RBF)
     -> fills the Q column of Supplementary Table F.
"""
import sys, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from math import comb
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
from sklearn.metrics import roc_auc_score

from asqdt import config as cfg
from asqdt.scripts.run_v2_revision import load_or_build_mp
from asqdt.scripts.run_v2_honest import fresh_noms, ensure_tank_stats, kscore
from asqdt.evaluation.fastq import (gram_from_sv, mmd_u, batch_statevectors,
                                    nominal_quantile)

ATTACK_LABELS = {
    'fdi': 'FDI', 'command': 'Sinusoidal', 'ramp': 'Ramp', 'composite': 'Composite',
    'cstr_fdi': 'FDI', 'cstr_thermal': 'Thermal', 'cstr_concentration': 'Conc.',
    'tank_bias': 'Bias', 'tank_oscillation': 'Oscillation', 'tank_drift': 'Drift'}


def sign_test(diffs):
    n = len(diffs)
    k = max(sum(d > 0 for d in diffs), sum(d < 0 for d in diffs))
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def item_A(MP):
    print("\n" + "=" * 74)
    print(" ITEM A — seed-level Q vs RBF sign tests (Finding 1)")
    print("=" * 74)
    for bname in ('CNC', 'CSTR', 'Tank'):
        for a in MP[bname][cfg.SEEDS[0]]['fam']:
            q = [MP[bname][s]['fam'][a]['q']['mmd'] for s in cfg.SEEDS]
            r = [MP[bname][s]['fam'][a]['rbf']['mmd'] for s in cfg.SEEDS]
            d = [qi - ri for qi, ri in zip(q, r)]
            consistent = all(x > 0 for x in d) or all(x < 0 for x in d)
            direction = 'Q>RBF' if np.mean(d) > 0 else 'RBF>Q'
            p = sign_test(d) if consistent else None
            print(f"  {bname:5s} {ATTACK_LABELS[a]:12s} {direction}  "
                  f"sign-consistent={consistent}  p={p if p else 'n/a'}  "
                  f"per-seed diffs={[f'{x:+.4f}' for x in d]}")
    print("\n  -> In Finding 1: keep the thermal p=0.0625 sentence ONLY if the "
          "CSTR Thermal row above shows Q>RBF, consistent=True; otherwise "
          "delete the sign-test clause. Mention CNC consistency only if shown.")


def item_B(MP):
    print("\n" + "=" * 74)
    print(" ITEM B — concentration bands, k-bar, bias, invalid-threshold FPR")
    print("=" * 74)
    mp = MP['CNC'][42]; d = mp['data']
    F = (fresh_noms('CNC', 42, d, 30) - mp['mu']) @ mp['V']
    Xa = (np.nan_to_num(d['atk']['command']['windows']) - mp['mu']) @ mp['V']
    K_rr = mp['K_q'][np.triu_indices(len(mp['K_q']), 1)]
    K_rn = gram_from_sv(mp['S_nom'], mp['qcache'].sv(F)).ravel()
    K_ra = gram_from_sv(mp['S_nom'], mp['qcache'].sv(Xa)).ravel()
    kbar = K_rr.mean(); m = len(mp['K_q'])
    print(f"  [C1] = {K_rr.mean():.3f}\\pm{K_rr.std():.3f}")
    print(f"  [C2] = {K_rn.mean():.3f}\\pm{K_rn.std():.3f}")
    print(f"  [C3] = {K_ra.mean():.3f}\\pm{K_ra.std():.3f}   (sinusoidal attack)")
    print(f"  [K1] = {kbar:.2f}    [K2] = {(2/m)*(1-kbar):.3f}")
    fprs = []
    for seed in cfg.SEEDS:
        mps = MP['CNC'][seed]; ds = mps['data']
        Fs = (fresh_noms('CNC', seed, ds, 30) - mps['mu']) @ mps['V']
        # initial (invalid) threshold: reference windows scored vs full set incl. self
        s_in = kscore(mps['K_q'], mps['K_q'])  # includes self-similarity in cross term
        tau_old = nominal_quantile(s_in)
        s_fresh = kscore(mps['K_q'], gram_from_sv(mps['S_nom'], mps['qcache'].sv(Fs)))
        fprs.append(np.mean(s_fresh > tau_old))
    lo, hi = 100 * min(fprs), 100 * max(fprs)
    print(f"  [K3] = {lo:.0f}--{hi:.0f}   (fresh-nominal FPR %, per-seed: "
          f"{[f'{f:.2f}' for f in fprs]})")


def item_C(MP):
    print("\n" + "=" * 74)
    print(" ITEM C — encoding-scale probe (held-out AUC vs input scale c)")
    print("=" * 74)
    for bname, fam, tag in (('CNC', 'command', 'P1/P2'),
                            ('CSTR', 'cstr_thermal', 'P3/P4')):
        mp = MP[bname][42]; d = mp['data']
        F = (fresh_noms(bname, 42, d, 30) - mp['mu']) @ mp['V']
        Xa = (np.nan_to_num(d['atk'][fam]['windows']) - mp['mu']) @ mp['V']
        y = np.concatenate([np.zeros(len(F)), np.ones(len(Xa))])
        row = []
        for c in (1.0, 0.5, 0.25, 0.1):
            Sn = batch_statevectors(mp['qc'], mp['Xn_p'] * c, mp['theta'])
            K = gram_from_sv(Sn); np.fill_diagonal(K, 1.0)
            def sc(X):
                S = batch_statevectors(mp['qc'], X * c, mp['theta'])
                return kscore(K, gram_from_sv(Sn, S))
            row.append((c, roc_auc_score(y, np.concatenate([sc(F), sc(Xa)]))))
        print(f"  {bname}/{fam} [{tag}]: " +
              "  ".join(f"c={c}: AUC={v:.2f}" for c, v in row))
    print("\n  -> [P1]=CNC AUC at c=1, [P2]=CNC AUC at c=0.1, "
          "[P3]/[P4]=CSTR thermal at c=1/0.1. Adjust the following sentence in "
          "the paper to match (partial recovery vs. no recovery).")


def item_D(MP):
    print("\n" + "=" * 74)
    print(" ITEM D — in-set-only batch MMD (Supplementary Table F rows)")
    print("=" * 74)
    for bname in ('CNC', 'CSTR', 'Tank'):
        P = cfg.ATTACK_SET[bname]
        rho, eps, emax = P['rho'], P['epsilon'], P['E_max']
        for a in MP[bname][cfg.SEEDS[0]]['data']['atk']:
            rows = {'q': [], 'r': [], 'n': []}
            for seed in cfg.SEEDS:
                mp = MP[bname][seed]; ai = mp['data']['atk'][a]
                ok = np.array([(ai['impacts'][e] >= rho) and
                               (ai['legacy_responses'][e] <= eps) and
                               (ai['energies'][e] <= emax)
                               for e in range(len(ai['impacts']))])
                rows['n'].append(int(ok.sum()))
                if ok.sum() < 2:
                    continue
                Xa = (np.nan_to_num(np.array(ai['windows'])[ok]) - mp['mu']) @ mp['V']
                Sa = mp['qcache'].sv(Xa)
                Kaa = gram_from_sv(Sa); np.fill_diagonal(Kaa, 1.0)
                rows['q'].append(mmd_u(mp['K_q'], Kaa, gram_from_sv(mp['S_nom'], Sa)))
                g = mp['gamma']; Xn = mp['Xn_p']
                rows['r'].append(mmd_u(sk_rbf(Xn, Xn, g), sk_rbf(Xa, Xa, g),
                                       sk_rbf(Xn, Xa, g)))
            f = (lambda v: f"${'+' if np.mean(v)>=0 else '-'}$"
                 f"{abs(np.mean(v)):.3f}$\\pm${np.std(v):.3f}") if rows['q'] else None
            star = r"$^{\ast}$" if len(rows['q']) < 5 else ""
            print(f"  {bname} & {ATTACK_LABELS[a]}{star} & {np.mean(rows['n']):.1f} & "
                  f"{f(rows['q'])} & {f(rows['r'])} \\\\   ({len(rows['q'])}/5 seeds)")


def main():
    MP = load_or_build_mp(set())
    ensure_tank_stats(MP)
    items = sys.argv[1:] or ['A', 'B', 'C', 'D']
    if 'A' in items: item_A(MP)
    if 'B' in items: item_B(MP)
    if 'C' in items: item_C(MP)
    if 'D' in items: item_D(MP)
    print("\nDONE. Paste the bracketed values into main_v2.tex (search for "
          "'%% FILL' / '%% VERIFY') and supplement_v2.tex (Supp F, Supp L).")


if __name__ == '__main__':
    main()
