#!/usr/bin/env python3
"""
Honest-protocol per-window evaluation (v2 revision diagnostic).

Fixes two evaluation flaws in the per-window detection protocol used by
run_tifs_v2.py (and hence in the published Table VI numbers):

  (1) SELF-INCLUSION BIAS: the nominal-quantile threshold was calibrated by
      scoring the m reference windows against a reference set INCLUDING
      themselves; the self-similarity term k(x,x)=1 deflates in-sample scores,
      so every out-of-sample window (nominal or attacked) exceeds the
      threshold. Fixed with leave-one-out (LOO) calibration.
  (2) IN-SAMPLE NEGATIVES: F1 counted the same reference windows as negatives.
      Fixed by evaluating against 30 FRESH held-out nominal episodes.

Under this protocol the oracle quantum kernel's single-window score has
held-out AUC ~= 0.5 on every benchmark and family (no discriminative signal;
see the kernel-concentration diagnostic in the findings report), while RBF
retains signal exactly where its batch MMD is positive.

Usage: python -m asqdt.scripts.eval_honest_protocol   (requires mp_cache.pkl
from run_v2_revision, or builds it).
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asqdt import config as cfg
from asqdt.scripts.run_v2_revision import load_or_build_mp, RESULTS
from asqdt.evaluation.fastq import gram_from_sv, nominal_quantile
from asqdt.cpms.plant import CPMSPlant, ObserverDT
from asqdt.cpms.data import run_episode, extract_one_window
from asqdt.cpms import cstr as C, tank as T
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
from sklearn.metrics import f1_score, roc_auc_score


def loo_scores(K_ref):
    m = len(K_ref); out = np.zeros(m)
    for j in range(m):
        idx = [i for i in range(m) if i != j]
        Ks = K_ref[np.ix_(idx, idx)]
        mr = (np.sum(Ks) - np.trace(Ks)) / ((m - 1) * (m - 2))
        out[j] = mr + 1.0 - 2.0 * np.mean(K_ref[idx, j])
    return out


def oos_scores(K_ref, K_r_test):
    m = len(K_ref)
    mr = (np.sum(K_ref) - np.trace(K_ref)) / (m * (m - 1))
    return mr + 1.0 - 2.0 * K_r_test.mean(axis=0)


def fresh_noms(bname, seed, d, n=30):
    W_L = d['W_L']; out = []
    for ep in range(n):
        if bname == 'CNC':
            p = CPMSPlant(seed=seed + 9000 + ep)
            t = ObserverDT(mismatch_frac=0.05, seed=seed + 1)
            dd = run_episode(p, t, p.reference(cfg.HORIZON), sensor_mask=np.ones(6))
            w = extract_one_window(dd['residuals'], W_L)
        elif bname == 'CSTR':
            p = C.CSTRPlant(seed=seed + 9000 + ep)
            t = C.CSTRObserverDT(mismatch_frac=0.05, seed=seed + 1)
            dd = C.run_cstr_episode(p, t, p.reference(120), sensor_mask=np.ones(3))
            k0 = 120 // 2 - W_L // 2
            w = dd['residuals'][k0:k0 + W_L].flatten()
        else:
            p = T.CoupledTankPlant(seed=seed + 9000 + ep)
            t = T.TankObserverDT(mismatch_frac=0.05, seed=seed + 1)
            dd = T.run_tank_episode(p, t, p.reference(120), sensor_mask=np.ones(2))
            k0 = 120 // 2 - W_L // 2
            w = np.nan_to_num(dd['residuals'][k0:k0 + W_L].flatten())
        out.append((w - d['nom_mean']) / d['nom_std'])
    return np.nan_to_num(np.array(out))


def main():
    MP = load_or_build_mp(set())
    for seed in cfg.SEEDS:      # Tank data dict lacks nom stats; recompute
        d = MP['Tank'][seed]['data']
        if 'nom_mean' not in d:
            nw = []
            for ep in range(20):
                p = T.CoupledTankPlant(seed=seed + 100 + ep)
                t = T.TankObserverDT(mismatch_frac=0.05, seed=seed + 1)
                dd = T.run_tank_episode(p, t, p.reference(120), sensor_mask=np.ones(2))
                k0 = 120 // 2 - 4
                nw.append(dd['residuals'][k0:k0 + 8].flatten())
            nw = np.nan_to_num(np.array(nw))
            d['nom_mean'], d['nom_std'] = nw.mean(0), nw.std(0) + 1e-8

    print("HONEST PROTOCOL (LOO threshold + 30 held-out nominal negatives), 5-seed means")
    print(f"{'bench':5s} {'attack':20s} {'Q F1':>10s} {'Q AUC':>7s} {'RBF F1':>10s} {'RBF AUC':>8s}")
    RES = {}
    for bname in ('CNC', 'CSTR', 'Tank'):
        fams = list(MP[bname][cfg.SEEDS[0]]['data']['atk'].keys())
        acc = {a: {'qf': [], 'qa': [], 'rf': [], 'ra': []} for a in fams}
        for seed in cfg.SEEDS:
            mp = MP[bname][seed]; d = mp['data']
            Xf = (fresh_noms(bname, seed, d) - mp['mu']) @ mp['V']
            Xn_p = mp['Xn_p']; Sf = mp['qcache'].sv(Xf)
            Kq = mp['K_q']; Kr = sk_rbf(Xn_p, Xn_p, mp['gamma'])
            tau_q = nominal_quantile(loo_scores(Kq))
            tau_r = nominal_quantile(loo_scores(Kr))
            sq_ho = oos_scores(Kq, gram_from_sv(mp['S_nom'], Sf))
            sr_ho = oos_scores(Kr, sk_rbf(Xn_p, Xf, mp['gamma']))
            for a in fams:
                Xa = (np.nan_to_num(d['atk'][a]['windows']) - mp['mu']) @ mp['V']
                sq_a = oos_scores(Kq, gram_from_sv(mp['S_nom'], mp['qcache'].sv(Xa)))
                sr_a = oos_scores(Kr, sk_rbf(Xn_p, Xa, mp['gamma']))
                y = np.concatenate([np.zeros(len(Xf)), np.ones(len(Xa))])
                acc[a]['qf'].append(f1_score(y, (np.concatenate([sq_ho, sq_a]) > tau_q).astype(int), zero_division=0))
                acc[a]['qa'].append(roc_auc_score(y, np.concatenate([sq_ho, sq_a])))
                acc[a]['rf'].append(f1_score(y, (np.concatenate([sr_ho, sr_a]) > tau_r).astype(int), zero_division=0))
                acc[a]['ra'].append(roc_auc_score(y, np.concatenate([sr_ho, sr_a])))
        for a in fams:
            v = acc[a]
            print(f"{bname:5s} {a:20s} {np.mean(v['qf']):.2f}±{np.std(v['qf']):.2f} "
                  f"{np.mean(v['qa']):>7.2f} {np.mean(v['rf']):.2f}±{np.std(v['rf']):.2f} "
                  f"{np.mean(v['ra']):>8.2f}")
        RES[bname] = {a: {k: [float(x) for x in vv] for k, vv in acc[a].items()} for a in fams}
    with open(os.path.join(RESULTS, 'v2_honest_protocol.json'), 'w') as f:
        json.dump(RES, f, indent=2)
    print(f"saved {os.path.join(RESULTS, 'v2_honest_protocol.json')}")


if __name__ == '__main__':
    main()
