#!/usr/bin/env python3
"""
Honest-protocol v2 runs: fills the corrected paper tables.

Protocol (uniform for ALL methods; Supplementary N/L corrected):
  - Reference set: the m=20 nominal windows (models/kernels fitted here).
  - Calibration set: 30 FRESH nominal episodes -> threshold = 95% quantile
    of their scores (fixes the self-inclusion bias of the v1 protocol).
  - Test set: 30 further FRESH nominal episodes (negatives) + the 10 attack
    windows (positives). Report F1 at the calibrated threshold and AUC.

Stages:
  H_TAB8   Table VIII (Q, RBF, Laplacian, tuned OC-SVM/IForest, AE, default
           OC-SVM, residual-norm) + artifact-diagnosis table numbers +
           Spearman(batch MMD, held-out AUC).
  H_C_CNC / H_C_CSTR   Table IX: Q and RBF batch MMD per sensor config.
  H_D_CNC / H_D_CSTR / H_D_Tank   Table X: adaptive attacks + honest AUC.
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf, laplacian_kernel as sk_lap
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import OneClassSVM

from asqdt import config as cfg
from asqdt.scripts.run_v2_revision import (load_or_build_mp, RESULTS, save_stage,
                                           gen_bench, ADAPTERS, strongest_inset_library, ms)
from asqdt.evaluation.fastq import (gram_from_sv, mmd_u, pca_fit, QKernelCache,
                                    nominal_quantile)
from asqdt.evaluation.baselines_v2 import (fit_tuned_ocsvm, fit_tuned_iforest,
                                           WindowAutoencoder)
from asqdt.attacks_adaptive import AdaptiveAttacker
from asqdt.quantum.kernel import QuantumCircuit
from asqdt.cpms.plant import CPMSPlant, ObserverDT
from asqdt.cpms.data import run_episode, extract_one_window
from asqdt.cpms import cstr as C, tank as T

ALPHA = cfg.DELTA
ATTACK_LABELS = {
    'fdi': 'FDI', 'command': 'Sinusoidal', 'ramp': 'Ramp', 'composite': 'Composite',
    'cstr_fdi': 'FDI', 'cstr_thermal': 'Thermal', 'cstr_concentration': 'Conc.',
    'tank_bias': 'Bias', 'tank_oscillation': 'Oscill.', 'tank_drift': 'Drift'}


def ensure_tank_stats(MP):
    for seed in cfg.SEEDS:
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


def fresh_noms(bname, seed, d, n=60, base=9000, mask=None):
    """n fresh nominal episodes; standardized with the training stats."""
    W_L = d['W_L']; out = []
    for ep in range(n):
        s = seed + base + ep
        if bname == 'CNC':
            m = np.ones(6) if mask is None else mask
            p = CPMSPlant(seed=s); t = ObserverDT(mismatch_frac=0.05, seed=seed + 1)
            dd = run_episode(p, t, p.reference(cfg.HORIZON), sensor_mask=m)
            w = extract_one_window(dd['residuals'], W_L)
        elif bname == 'CSTR':
            m = np.ones(3) if mask is None else mask
            p = C.CSTRPlant(seed=s); t = C.CSTRObserverDT(mismatch_frac=0.05, seed=seed + 1)
            dd = C.run_cstr_episode(p, t, p.reference(120), sensor_mask=m)
            k0 = 120 // 2 - W_L // 2
            w = dd['residuals'][k0:k0 + W_L].flatten()
        else:
            m = np.ones(2) if mask is None else mask
            p = T.CoupledTankPlant(seed=s); t = T.TankObserverDT(mismatch_frac=0.05, seed=seed + 1)
            dd = T.run_tank_episode(p, t, p.reference(120), sensor_mask=m)
            k0 = 120 // 2 - W_L // 2
            w = np.nan_to_num(dd['residuals'][k0:k0 + W_L].flatten())
        out.append((w - d['nom_mean']) / d['nom_std'])
    return np.nan_to_num(np.array(out))


def kscore(K_ref, K_r_test):
    """S(x) for out-of-sample x from precomputed cross gram (self-sim = 1)."""
    m = len(K_ref)
    mr = (np.sum(K_ref) - np.trace(K_ref)) / (m * (m - 1))
    return mr + 1.0 - 2.0 * K_r_test.mean(axis=0)


def honest_methods(mp, Xcal, Xtest):
    """Build {name: (score_fn, tau)} calibrated on Xcal, for one (bench, seed)."""
    Xn_p, gamma = mp['Xn_p'], mp['gamma']
    qcache, S_nom, K_q = mp['qcache'], mp['S_nom'], mp['K_q']
    Kr = sk_rbf(Xn_p, Xn_p, gamma)
    Kl = sk_lap(Xn_p, Xn_p, gamma)
    Xa_tune = mp['Xa_pool'][:len(mp['Xa_pool']) // 2]
    oc_t, _, _ = fit_tuned_ocsvm(Xn_p, Xa_tune)
    if_t, _, _ = fit_tuned_iforest(Xn_p, Xa_tune, seed=42)
    ae = WindowAutoencoder(seed=0).fit(Xn_p)
    oc_d = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale').fit(Xn_p)

    fns = {
        'q':    lambda X: kscore(K_q, gram_from_sv(S_nom, qcache.sv(X))),
        'rbf':  lambda X: kscore(Kr, sk_rbf(Xn_p, X, gamma)),
        'lap':  lambda X: kscore(Kl, sk_lap(Xn_p, X, gamma)),
        'ocsvm_t': lambda X: -oc_t.decision_function(X),
        'iforest_t': lambda X: -if_t.decision_function(X),
        'ae':   ae.score,
        'ocsvm_d': lambda X: -oc_d.decision_function(X),
        'resnorm': lambda X: np.linalg.norm(X, axis=1),
    }
    out = {}
    for name, fn in fns.items():
        tau = np.percentile(fn(Xcal), 100 * (1 - ALPHA))
        out[name] = (fn, tau, fn(Xtest))   # cache test-negative scores
    return out


METHODS = ['q', 'rbf', 'lap', 'ocsvm_t', 'iforest_t', 'ae', 'ocsvm_d', 'resnorm']


def stage_tab8(MP):
    print("\n" + "=" * 78)
    print(" H_TAB8: HONEST PER-WINDOW DETECTION (Table VIII + artifact table)")
    print("=" * 78)
    res = {b: {} for b in ('CNC', 'CSTR', 'Tank')}
    pooled = {'mmd': [], 'auc': []}
    for bname in ('CNC', 'CSTR', 'Tank'):
        fams = list(MP[bname][cfg.SEEDS[0]]['data']['atk'].keys())
        acc = {a: {mname: {'f1': [], 'auc': [], 'fpr': []} for mname in METHODS} for a in fams}
        for seed in cfg.SEEDS:
            mp = MP[bname][seed]; d = mp['data']
            F = (fresh_noms(bname, seed, d, 60) - mp['mu']) @ mp['V']
            Xcal, Xtest = F[:30], F[30:]
            H = honest_methods(mp, Xcal, Xtest)
            for a in fams:
                Xa = (np.nan_to_num(d['atk'][a]['windows']) - mp['mu']) @ mp['V']
                y = np.concatenate([np.zeros(30), np.ones(len(Xa))])
                for mname in METHODS:
                    fn, tau, s_neg = H[mname]
                    s = np.concatenate([s_neg, fn(Xa)])
                    acc[a][mname]['f1'].append(
                        f1_score(y, (s > tau).astype(int), zero_division=0))
                    acc[a][mname]['auc'].append(roc_auc_score(y, s))
                    acc[a][mname]['fpr'].append(float(np.mean(s_neg > tau)))
                pooled['mmd'].append(mp['fam'][a]['q']['mmd'])
                pooled['auc'].append(acc[a]['q']['auc'][-1])
                pooled['mmd'].append(mp['fam'][a]['rbf']['mmd'])
                pooled['auc'].append(acc[a]['rbf']['auc'][-1])
            print(f"  {bname} seed={seed} done", flush=True)
        for a in fams:
            res[bname][a] = {mname: {k: (float(np.mean(v[k])), float(np.std(v[k])))
                                     for k in ('f1', 'auc', 'fpr')}
                             for mname, v in acc[a].items()}
    rho, pv = spearmanr(pooled['mmd'], pooled['auc'])
    res['spearman_batchmmd_vs_honestauc'] = (float(rho), float(pv), len(pooled['mmd']))
    print(f"\n  Spearman rho_S(batch MMD, honest AUC) = {rho:+.3f} (p={pv:.1e}, "
          f"n={len(pooled['mmd'])})")

    print("\n  LaTeX Table VIII rows (F1 honest; order: Q, RBF, Lap, OCSVMt, IFt, AE, OCSVMd, ResN):")
    for bname in ('CNC', 'CSTR', 'Tank'):
        for a in res[bname]:
            r = res[bname][a]
            cells = " & ".join(f"{r[mn]['f1'][0]:.2f}$\\pm${r[mn]['f1'][1]:.2f}"
                               for mn in METHODS)
            print(f"  {bname} & {ATTACK_LABELS[a]} & {cells} \\\\")
    print("\n  LaTeX artifact-table rows (Q insampleF1 / hoAUC / honestF1, then RBF):")
    for bname in ('CNC', 'CSTR', 'Tank'):
        for a in res[bname]:
            r = res[bname][a]
            ins_q = np.mean([MP[bname][s]['fam'][a]['q']['f1'] for s in cfg.SEEDS])
            ins_r = np.mean([MP[bname][s]['fam'][a]['rbf']['f1'] for s in cfg.SEEDS])
            print(f"  {bname} & {ATTACK_LABELS[a]} & {ins_q:.2f} & {r['q']['auc'][0]:.2f} & "
                  f"{r['q']['f1'][0]:.2f} & {ins_r:.2f} & {r['rbf']['auc'][0]:.2f} & "
                  f"{r['rbf']['f1'][0]:.2f} \\\\")
    save_stage('honest_tab8', res)


def eval_mask2(bname, seed, mask, theta, gamma):
    """Q and RBF batch MMD under a sensor mask (frozen theta, gamma)."""
    d = gen_bench(bname, seed, sensor_mask=mask)
    Xn, atk = d['nom'], d['atk']
    mu, V = pca_fit(Xn, cfg.N_QUBITS * cfg.CIRCUIT_DEPTH)
    Xn_p = (Xn - mu) @ V
    qc = QuantumCircuit(cfg.N_QUBITS, cfg.CIRCUIT_DEPTH)
    qcache = QKernelCache(qc, theta)
    S_nom = qcache.sv(Xn_p)
    K_q = gram_from_sv(S_nom); np.fill_diagonal(K_q, 1.0)
    Kr = sk_rbf(Xn_p, Xn_p, gamma)
    qs, rs = [], []
    for aname, ainfo in atk.items():
        Xa = (np.nan_to_num(ainfo['windows']) - mu) @ V
        Sa = qcache.sv(Xa)
        Kaa = gram_from_sv(Sa); np.fill_diagonal(Kaa, 1.0)
        qs.append(mmd_u(K_q, Kaa, gram_from_sv(S_nom, Sa)))
        rs.append(mmd_u(Kr, sk_rbf(Xa, Xa, gamma), sk_rbf(Xn_p, Xa, gamma)))
    return float(np.mean(qs)), float(np.mean(rs))


def stage_C(MP, bname, budget):
    from asqdt.quantum.surrogate_variants import SurrogateVariant
    from asqdt.codesign.sensors_v2 import (separation_matrices_v2, greedy_logdet,
                                           top_variance_sensors, mask_from)
    from itertools import combinations
    print("\n" + "=" * 78)
    print(f" H_C: SENSOR CO-DESIGN, Q & RBF BATCH MMD ({bname}, budget {budget})")
    print("=" * 78)
    p = MP[bname][cfg.SEEDS[0]]['data']['p']
    W_L = MP[bname][cfg.SEEDS[0]]['data']['W_L']
    res = {c: {'q': [], 'r': []} for c in ('greedy', 'random', 'topvar', 'all')}
    sel_log = {}
    for seed in cfg.SEEDS:
        mp = MP[bname][seed]
        train_X = np.vstack([mp['Xn_p'], mp['Xa_pool'][:len(mp['Xn_p'])]])
        v0 = SurrogateVariant(15, (64, 32, 16), 500, normalized=False, seed=0)
        v0.fit(train_X, mp['qcache'].gram(train_X))
        raw_atk = np.nan_to_num(np.vstack([a['windows'] for a in mp['data']['atk'].values()]))
        M = separation_matrices_v2(raw_atk, v0.embed, mp['mu'], mp['V'], p, W_L)
        g_sel = greedy_logdet(M, budget)
        tv_sel = top_variance_sensors(mp['data']['nom_std'], p, W_L, budget)
        sel_log[seed] = dict(greedy=g_sel, topvar=tv_sel)
        for cname, sel in (('greedy', g_sel), ('topvar', tv_sel)):
            q, r = eval_mask2(bname, seed, mask_from(sel, p), mp['theta'], mp['gamma'])
            res[cname]['q'].append(q); res[cname]['r'].append(r)
        q, r = eval_mask2(bname, seed, np.ones(p), mp['theta'], mp['gamma'])
        res['all']['q'].append(q); res['all']['r'].append(r)
        rng = np.random.RandomState(seed + 77)
        all_masks = list(combinations(range(p), budget))
        draws = all_masks if len(all_masks) <= 10 else \
            [all_masks[i] for i in rng.choice(len(all_masks), 10, replace=False)]
        rq, rr = [], []
        for dsel in draws:
            q, r = eval_mask2(bname, seed, mask_from(dsel, p), mp['theta'], mp['gamma'])
            rq.append(q); rr.append(r)
        res['random']['q'].append(np.mean(rq)); res['random']['r'].append(np.mean(rr))
        print(f"  {bname} seed={seed}: greedy={g_sel} Qmmd={res['greedy']['q'][-1]:+.4f} | "
              f"topvar={tv_sel} | all Qmmd={res['all']['q'][-1]:+.4f}", flush=True)
    print(f"  LaTeX ({bname} {budget}/{p}): config & Q MMD & RBF MMD")
    out = {'budget': budget, 'selected': {str(k): v for k, v in sel_log.items()}}
    for c, lab in (('greedy', 'Greedy AS-QDT'), ('random', 'Random'),
                   ('topvar', 'Top-variance'), ('all', 'All sensors')):
        v = res[c]
        out[c] = dict(q=f"{np.mean(v['q']):+.4f}$\\pm${np.std(v['q']):.4f}",
                      r=f"{np.mean(v['r']):+.4f}$\\pm${np.std(v['r']):.4f}")
        print(f"    {lab} & {out[c]['q']} & {out[c]['r']} \\\\")
    save_stage(f'honest_C_{bname}_b{budget}', out)


def stage_D(MP, bname):
    print("\n" + "=" * 78)
    print(f" H_D: ADAPTIVE ATTACKS + HONEST AUC ({bname})")
    print("=" * 78)
    params = cfg.ATTACK_SET[bname]
    keys = ('I', 'D', 'mmd', 'aucq', 'aucr', 'inset')
    rows = {t: {k: [] for k in keys} for t in ('lib', 'ada')}
    for seed in cfg.SEEDS:
        mp = MP[bname][seed]; d = mp['data']
        F = (fresh_noms(bname, seed, d, 60) - mp['mu']) @ mp['V']
        Xcal, Xtest = F[:30], F[30:]
        Xn_p, gamma = mp['Xn_p'], mp['gamma']
        Kr = sk_rbf(Xn_p, Xn_p, gamma)
        sq_neg = kscore(mp['K_q'], gram_from_sv(mp['S_nom'], mp['qcache'].sv(Xtest)))
        sr_neg = kscore(Kr, sk_rbf(Xn_p, Xtest, gamma))
        rollout, gens, shape = ADAPTERS[bname](seed, mp)
        a_lib, lib_name, lib_ep = strongest_inset_library(bname, seed, mp, gens, shape)
        att = AdaptiveAttacker(rollout, cfg.N_EPISODES_ATK, params,
                               mp['mu'], mp['V'], mp['qcache'], mp['S_nom'], mp['K_q'])
        ev_lib = att.evaluate(a_lib)
        a_ada, ev_ada = att.optimize(a_lib, n_iters=80, seed=seed)
        for tag, ev in (('lib', ev_lib), ('ada', ev_ada)):
            Xa = ev['Xa']
            y = np.concatenate([np.zeros(30), np.ones(len(Xa))])
            sq = np.concatenate([sq_neg, kscore(mp['K_q'],
                                                gram_from_sv(mp['S_nom'], mp['qcache'].sv(Xa)))])
            sr = np.concatenate([sr_neg, kscore(Kr, sk_rbf(Xn_p, Xa, gamma))])
            rows[tag]['I'].append(np.mean(ev['impacts']))
            rows[tag]['D'].append(np.mean(ev['legacies']))
            rows[tag]['mmd'].append(ev['mmd'])
            rows[tag]['aucq'].append(roc_auc_score(y, sq))
            rows[tag]['aucr'].append(roc_auc_score(y, sr))
            rows[tag]['inset'].append(ev['in_set'])
        print(f"  {bname} seed={seed} [init={lib_name}/ep{lib_ep}] "
              f"lib mmd={ev_lib['mmd']:+.4f} | ada mmd={ev_ada['mmd']:+.4f} "
              f"aucq={rows['ada']['aucq'][-1]:.2f} aucr={rows['ada']['aucr'][-1]:.2f} "
              f"in={ev_ada['in_set']}/10", flush=True)
    out = {}
    for tag, lab in (('lib', 'Library'), ('ada', 'Adaptive')):
        v = rows[tag]
        out[tag] = dict(I=f"{np.mean(v['I']):.2f}", D=f"{np.mean(v['D']):.1f}",
                        mmd=f"{np.mean(v['mmd']):+.4f}$\\pm${np.std(v['mmd']):.4f}",
                        aucq=ms(v['aucq']), aucr=ms(v['aucr']),
                        inset=f"{np.mean(v['inset']):.1f}/10")
        print(f"  LaTeX: {bname} & {lab} & {out[tag]['I']} & {out[tag]['D']} & "
              f"{np.mean(v['mmd']):+.4f} & {out[tag]['aucq']} & {out[tag]['aucr']} & "
              f"{out[tag]['inset']} \\\\")
    save_stage(f'honest_D_{bname}', out)


def main():
    stages = sys.argv[1:] or ['H_TAB8']
    MP = load_or_build_mp(set())
    ensure_tank_stats(MP)
    if 'H_TAB8' in stages: stage_tab8(MP)
    if 'H_C_CNC' in stages: stage_C(MP, 'CNC', 4)
    if 'H_C_CNC2' in stages: stage_C(MP, 'CNC', 2)
    if 'H_C_CSTR' in stages: stage_C(MP, 'CSTR', 2)
    if 'H_D_CNC' in stages: stage_D(MP, 'CNC')
    if 'H_D_CSTR' in stages: stage_D(MP, 'CSTR')
    if 'H_D_Tank' in stages: stage_D(MP, 'Tank')
    print("\nHONEST STAGES DONE:", stages)


if __name__ == '__main__':
    main()
