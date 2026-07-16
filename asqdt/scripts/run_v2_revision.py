#!/usr/bin/env python3
"""
AS-QDT Version-2 revision experiments (fills the TODO-DATA placeholders).

  EXP A  Margin--detection link diagnostic     -> main Table V   (Supp. L)
  EXP B  Tuned OC-SVM / IForest + autoencoder  -> main Table VIII (Supp. N)
  EXP C  Multi-seed budgeted sensor co-design  -> main Table IX  (Supp. P)
  EXP D  Adaptive-attack stress test           -> main Table X   (Supp. O)
  EXP E  Surrogate-improvement ablation        -> main Table VII (Supp. M)

The kernel design per (benchmark, seed) replicates run_tifs_v2.run_seed
bit-for-bit (same RNG consumption), so Q-Direct/RBF MMD and F1 values should
reproduce the published Tables IV/VI -- printed as a sanity check.
"""
import sys, os, time, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf, laplacian_kernel as sk_lap

from asqdt import config as cfg
from asqdt.quantum.kernel import QuantumCircuit
from asqdt.quantum.surrogate_variants import SurrogateVariant
from asqdt.evaluation.fastq import (batch_statevectors, gram_from_sv, QKernelCache,
                                    mmd_u, pw_scores_from_grams, detect_f1, safe_auc,
                                    nominal_quantile, worst_window_gap, linear_cka,
                                    pca_fit, design_quantum_kernel, rbf_bandwidth)
from asqdt.evaluation.baselines_v2 import (fit_tuned_ocsvm, fit_tuned_iforest,
                                           WindowAutoencoder, eval_scored_baseline)
from asqdt.codesign.sensors_v2 import (separation_matrices_v2, greedy_logdet,
                                       top_variance_sensors, mask_from)
from asqdt.attacks_adaptive import AdaptiveAttacker

from asqdt.cpms.data import generate_dataset, run_episode, extract_one_window
from asqdt.cpms.plant import (CPMSPlant, ObserverDT, ATTACK_GENERATORS,
                              legacy_detector_response, compute_impact)
from asqdt.cpms import cstr as C
from asqdt.cpms import tank as T

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS, exist_ok=True)
TD = cfg.N_QUBITS * cfg.CIRCUIT_DEPTH   # PCA dim d = 12
ALPHA = cfg.DELTA


# ══════════════════════════════════════════════════════════════════
#  Shared per-(benchmark, seed) pipeline
# ══════════════════════════════════════════════════════════════════

def gen_bench(bname, seed, sensor_mask=None):
    if bname == 'CNC':
        out = generate_dataset(mismatch_frac=0.05, seed=seed, W_L=10, E_max=2.0,
                               legacy_method='chi_sq', sensor_mask=sensor_mask)
        nom, atk, Sri, nm, ns = out
        return dict(nom=nom, atk=atk, Sigma_r_inv=Sri, nom_mean=nm, nom_std=ns, W_L=10, p=6)
    if bname == 'CSTR':
        nom, atk, nm, ns = C.generate_cstr_dataset(mismatch_frac=0.05, n_nom=20, n_atk=10,
                                                   H=120, seed=seed,
                                                   sensor_mask=sensor_mask, return_stats=True)
        return dict(nom=nom, atk=atk, nom_mean=nm, nom_std=ns, W_L=8, p=3)
    nom, atk = T.generate_tank_dataset(mismatch_frac=0.05, n_nom=20, n_atk=10, H=120, seed=seed)
    return dict(nom=nom, atk=atk, W_L=8, p=2)


def kernel_scores(K_ref, K_rt_all, ktt_all):
    return pw_scores_from_grams(K_ref, ktt_all, K_rt_all)


def exact_kernel_eval(Xn_p, Xa_t, qcache, S_nom, K_q, gamma):
    """Batch MMDs + per-window (score, f1, auc, dmin) for Q-Direct/RBF/Laplacian."""
    m, n = len(Xn_p), len(Xa_t)
    y = np.concatenate([np.zeros(m), np.ones(n)])
    X_all = np.vstack([Xn_p, Xa_t])
    out = {}

    # Q-Direct
    Sa = qcache.sv(Xa_t)
    Kaa = gram_from_sv(Sa); np.fill_diagonal(Kaa, 1.0)
    Kna = gram_from_sv(S_nom, Sa)
    S_all = np.vstack([S_nom, Sa])
    Krt = gram_from_sv(S_nom, S_all)
    sc = kernel_scores(K_q, Krt, np.ones(m + n))
    out['q'] = dict(mmd=mmd_u(K_q, Kaa, Kna), scores=sc)

    # RBF / Laplacian
    for name, kf in (('rbf', sk_rbf), ('lap', sk_lap)):
        Kr = kf(Xn_p, Xn_p, gamma)
        Ka = kf(Xa_t, Xa_t, gamma)
        Kna_c = kf(Xn_p, Xa_t, gamma)
        Krt_c = kf(Xn_p, X_all, gamma)
        sc = kernel_scores(Kr, Krt_c, np.ones(m + n))
        out[name] = dict(mmd=mmd_u(Kr, Ka, Kna_c), scores=sc)

    for name in ('q', 'rbf', 'lap'):
        sc = out[name]['scores']
        nom_sc, atk_sc = sc[:m], sc[m:]
        tau = nominal_quantile(nom_sc, ALPHA)
        out[name].update(f1=detect_f1(y, sc, tau), auc=safe_auc(y, sc),
                         dmin=worst_window_gap(nom_sc, atk_sc, ALPHA), tau=tau)
    return out


def variant_pw_f1(variant, Xn_p, Xa_t):
    m, n = len(Xn_p), len(Xa_t)
    y = np.concatenate([np.zeros(m), np.ones(n)])
    X_all = np.vstack([Xn_p, Xa_t])
    K_ref = variant.kernel(Xn_p)
    K_rt = variant.kernel(Xn_p, X_all)
    ktt = np.array([variant.kernel(X_all[i:i+1])[0, 0] for i in range(m + n)])
    sc = kernel_scores(K_ref, K_rt, ktt)
    tau = nominal_quantile(sc[:m], ALPHA)
    return detect_f1(y, sc, tau)


def main_pass(bname, seed):
    """Design + evaluate one (benchmark, seed); returns everything downstream."""
    d = gen_bench(bname, seed)
    Xn, atk = d['nom'], d['atk']
    mu, V = pca_fit(Xn, TD)
    Xn_p = (Xn - mu) @ V
    Xa_pool = (np.nan_to_num(np.vstack([a['windows'] for a in atk.values()])) - mu) @ V

    qc = QuantumCircuit(cfg.N_QUBITS, cfg.CIRCUIT_DEPTH)
    theta, margins = design_quantum_kernel(qc, Xn_p, Xa_pool, seed)
    gamma = rbf_bandwidth(Xn_p, Xa_pool[:len(Xa_pool) // 2])
    qcache = QKernelCache(qc, theta)
    S_nom = qcache.sv(Xn_p)
    K_q = gram_from_sv(S_nom); np.fill_diagonal(K_q, 1.0)

    # Tuned baselines fitted once per (bench, seed) on the tune split
    Xa_tune = Xa_pool[:len(Xa_pool) // 2]
    oc_t, oc_params, _ = fit_tuned_ocsvm(Xn_p, Xa_tune)
    if_t, if_params, _ = fit_tuned_iforest(Xn_p, Xa_tune, seed=42)
    ae = WindowAutoencoder(seed=0).fit(Xn_p)

    fam = {}
    for aname, ainfo in atk.items():
        Xa_t = (np.nan_to_num(ainfo['windows']) - mu) @ V
        r = exact_kernel_eval(Xn_p, Xa_t, qcache, S_nom, K_q, gamma)
        r['ocsvm_t'] = eval_scored_baseline(lambda X: -oc_t.decision_function(X), Xn_p, Xa_t, ALPHA)
        r['iforest_t'] = eval_scored_baseline(lambda X: -if_t.decision_function(X), Xn_p, Xa_t, ALPHA)
        r['ae'] = eval_scored_baseline(ae.score, Xn_p, Xa_t, ALPHA)
        fam[aname] = r

    return dict(data=d, mu=mu, V=V, Xn_p=Xn_p, Xa_pool=Xa_pool, qc=qc, theta=theta,
                qcache=qcache, S_nom=S_nom, K_q=K_q, gamma=gamma, fam=fam,
                tuned_params=dict(ocsvm=oc_params, iforest={k: str(v) for k, v in if_params.items()}))


# ══════════════════════════════════════════════════════════════════
#  EXP C: sensor study helpers
# ══════════════════════════════════════════════════════════════════

def eval_mask(bname, seed, mask, theta):
    """Regenerate data under mask, refit PCA, evaluate frozen theta.
    Returns (mean Q-MMD over families, mean Q F1 over families)."""
    d = gen_bench(bname, seed, sensor_mask=mask)
    Xn, atk = d['nom'], d['atk']
    mu, V = pca_fit(Xn, TD)
    Xn_p = (Xn - mu) @ V
    qc = QuantumCircuit(cfg.N_QUBITS, cfg.CIRCUIT_DEPTH)
    qcache = QKernelCache(qc, theta)
    S_nom = qcache.sv(Xn_p)
    K_q = gram_from_sv(S_nom); np.fill_diagonal(K_q, 1.0)
    m = len(Xn_p)
    mmds, f1s = [], []
    for aname, ainfo in atk.items():
        Xa_t = (np.nan_to_num(ainfo['windows']) - mu) @ V
        Sa = qcache.sv(Xa_t)
        Kaa = gram_from_sv(Sa); np.fill_diagonal(Kaa, 1.0)
        mmds.append(mmd_u(K_q, Kaa, gram_from_sv(S_nom, Sa)))
        X_all = np.vstack([Xn_p, Xa_t])
        Krt = gram_from_sv(S_nom, qcache.sv(X_all))
        sc = kernel_scores(K_q, Krt, np.ones(len(X_all)))
        tau = nominal_quantile(sc[:m], ALPHA)
        y = np.concatenate([np.zeros(m), np.ones(len(Xa_t))])
        f1s.append(detect_f1(y, sc, tau))
    return float(np.mean(mmds)), float(np.mean(f1s))


# ══════════════════════════════════════════════════════════════════
#  EXP D: adaptive-attack rollout adapters (fixed episode seeds)
# ══════════════════════════════════════════════════════════════════

def cnc_adapter(seed, mp):
    d = mp['data']; Sri = d['Sigma_r_inv']; nm, ns, W_L = d['nom_mean'], d['nom_std'], d['W_L']
    z_nom = []
    for ep in range(cfg.N_EPISODES_ATK):
        p = CPMSPlant(seed=seed + 200 + ep)
        t = ObserverDT(mismatch_frac=0.05, seed=seed + 1)
        ref = p.reference(cfg.HORIZON)
        z_nom.append(run_episode(p, t, ref, sensor_mask=np.ones(cfg.N_OUTPUT))['z_traj'])

    def rollout(a_seq, ep):
        p = CPMSPlant(seed=seed + 200 + ep)
        t = ObserverDT(mismatch_frac=0.05, seed=seed + 1)
        ref = p.reference(cfg.HORIZON)
        dd = run_episode(p, t, ref, a_seq, np.ones(cfg.N_OUTPUT))
        return dict(window_raw=(extract_one_window(dd['residuals'], W_L) - nm) / ns,
                    impact=compute_impact(dd['z_traj'], z_nom[ep]),
                    legacy=legacy_detector_response(dd['residuals'], method='chi_sq',
                                                    Sigma_r_inv=Sri))
    return rollout, ATTACK_GENERATORS, (cfg.HORIZON, cfg.N_ATTACK)


def cstr_adapter(seed, mp):
    d = mp['data']; nm, ns, W_L = d['nom_mean'], d['nom_std'], d['W_L']
    # mu_0 exactly as in the generator
    norms_all = []
    for ep in range(20):
        p = C.CSTRPlant(seed=seed + 100 + ep)
        t = C.CSTRObserverDT(mismatch_frac=0.05, seed=seed + 1)
        dd = C.run_cstr_episode(p, t, p.reference(120), sensor_mask=np.ones(3))
        norms_all.append(dd['residuals'])
    norms = np.linalg.norm(np.vstack(norms_all), axis=1)
    mu_0 = np.mean(norms) + 2 * np.std(norms)
    z_nom = []
    for ep in range(10):
        p = C.CSTRPlant(seed=seed + 200 + ep)
        t = C.CSTRObserverDT(mismatch_frac=0.05, seed=seed + 1)
        z_nom.append(C.run_cstr_episode(p, t, p.reference(120), sensor_mask=np.ones(3))['z_traj'])

    def rollout(a_seq, ep):
        p = C.CSTRPlant(seed=seed + 200 + ep)
        t = C.CSTRObserverDT(mismatch_frac=0.05, seed=seed + 1)
        dd = C.run_cstr_episode(p, t, p.reference(120), a_seq, np.ones(3))
        k0 = 120 // 2 - W_L // 2
        w = dd['residuals'][k0:k0 + W_L].flatten()
        return dict(window_raw=(w - nm) / ns,
                    impact=compute_impact(dd['z_traj'], z_nom[ep], W_z=C.W_Z_CSTR),
                    legacy=C.cusum_legacy(dd['residuals'], mu_0=mu_0))
    return rollout, C.CSTR_ATTACK_GENERATORS, (120, 3)


def tank_adapter(seed, mp):
    W_L = 8
    nom_w, all_res = [], []
    for ep in range(20):
        p = T.CoupledTankPlant(seed=seed + 100 + ep)
        t = T.TankObserverDT(mismatch_frac=0.05, seed=seed + 1)
        dd = T.run_tank_episode(p, t, p.reference(120), sensor_mask=np.ones(2))
        all_res.append(dd['residuals'])
        k0 = 120 // 2 - W_L // 2
        nom_w.append(dd['residuals'][k0:k0 + W_L].flatten())
    nom_w = np.nan_to_num(np.array(nom_w))
    nm, ns = nom_w.mean(0), nom_w.std(0) + 1e-8
    cov = np.cov(np.vstack(all_res).T) + 1e-6 * np.eye(2)
    Sig_inv = np.linalg.inv(cov)
    z_nom = []
    for ep in range(10):
        p = T.CoupledTankPlant(seed=seed + 200 + ep)
        t = T.TankObserverDT(mismatch_frac=0.05, seed=seed + 1)
        z_nom.append(T.run_tank_episode(p, t, p.reference(120), sensor_mask=np.ones(2))['z_traj'])

    def rollout(a_seq, ep):
        p = T.CoupledTankPlant(seed=seed + 200 + ep)
        t = T.TankObserverDT(mismatch_frac=0.05, seed=seed + 1)
        dd = T.run_tank_episode(p, t, p.reference(120), a_seq, np.ones(2))
        k0 = 120 // 2 - W_L // 2
        w = np.nan_to_num(dd['residuals'][k0:k0 + W_L].flatten())
        return dict(window_raw=(w - nm) / ns,
                    impact=compute_impact(dd['z_traj'], z_nom[ep]),
                    legacy=T.chi_squared_legacy(dd['residuals'], Sig_inv))
    return rollout, T.TANK_ATTACK_GENERATORS, (120, 2)


ADAPTERS = {'CNC': cnc_adapter, 'CSTR': cstr_adapter, 'Tank': tank_adapter}


def strongest_inset_library(bname, seed, mp, generators, shape):
    """Return the fixed a_seq of the strongest in-set library attack."""
    d = mp['data']; atk = d['atk']
    params = cfg.ATTACK_SET[bname]
    best = None
    for aname, ainfo in atk.items():
        for ep in range(len(ainfo['impacts'])):
            imp, leg = ainfo['impacts'][ep], ainfo['legacy_responses'][ep]
            if imp >= params['rho'] and leg <= params['epsilon']:
                if best is None or imp > best[0]:
                    best = (imp, aname, ep)
    if best is None:                       # no fully in-set episode: take max impact
        for aname, ainfo in atk.items():
            for ep in range(len(ainfo['impacts'])):
                imp = ainfo['impacts'][ep]
                if best is None or imp > best[0]:
                    best = (imp, aname, ep)
    _, aname, ep = best
    H, q = shape
    E_max = params['E_max']
    a_seq = generators[aname](H, np.random.RandomState(seed + 300 + ep), E_max=E_max)
    return a_seq, aname, ep


def pw_f1_for_fixed_attack(mp, Xa):
    """Per-window F1 (Q-Direct, RBF) of a fixed-attack window set vs nominal."""
    Xn_p, qcache, S_nom, K_q, gamma = mp['Xn_p'], mp['qcache'], mp['S_nom'], mp['K_q'], mp['gamma']
    m = len(Xn_p)
    y = np.concatenate([np.zeros(m), np.ones(len(Xa))])
    X_all = np.vstack([Xn_p, Xa])
    Krt = gram_from_sv(S_nom, qcache.sv(X_all))
    sc_q = kernel_scores(K_q, Krt, np.ones(len(X_all)))
    Kr = sk_rbf(Xn_p, Xn_p, gamma)
    sc_r = kernel_scores(Kr, sk_rbf(Xn_p, X_all, gamma), np.ones(len(X_all)))
    return (detect_f1(y, sc_q, nominal_quantile(sc_q[:m], ALPHA)),
            detect_f1(y, sc_r, nominal_quantile(sc_r[:m], ALPHA)))


# ══════════════════════════════════════════════════════════════════

def sanity_check_batch_statevectors():
    qc = QuantumCircuit(4, 3)
    rng = np.random.RandomState(7)
    X = rng.randn(5, 12)
    th = qc.initial_params(rng)
    S = batch_statevectors(qc, X, th)
    for i in range(5):
        ref = qc.statevector(X[i], th)
        assert np.allclose(S[i], ref, atol=1e-10), "batched statevector mismatch"
    K_fast = gram_from_sv(S)
    K_slow = qc.gram_matrix(X, th)
    np.fill_diagonal(K_fast, 1.0)
    assert np.allclose(K_fast, K_slow, atol=1e-10), "gram mismatch"
    print("[sanity] batched statevectors == QuantumCircuit.statevector  OK")


ATTACK_LABELS = {
    'fdi': 'FDI', 'command': 'Sinusoidal', 'ramp': 'Ramp', 'composite': 'Composite',
    'cstr_fdi': 'FDI', 'cstr_thermal': 'Thermal', 'cstr_concentration': 'Conc.',
    'tank_bias': 'Bias', 'tank_oscillation': 'Oscill.', 'tank_drift': 'Drift'}


def ms(vals):
    return f"{np.mean(vals):.2f}$\\pm${np.std(vals):.2f}"


MP_CACHE = os.path.join(RESULTS, 'mp_cache.pkl')


def load_or_build_mp(stages):
    import pickle
    if os.path.exists(MP_CACHE):
        with open(MP_CACHE, 'rb') as f:
            return pickle.load(f)
    print("\n[MAIN PASS] design + bridge diagnostic + tuned baselines")
    MP = {b: {} for b in ('CNC', 'CSTR', 'Tank')}
    for bname in ('CNC', 'CSTR', 'Tank'):
        for seed in cfg.SEEDS:
            tt = time.time()
            MP[bname][seed] = main_pass(bname, seed)
            r = MP[bname][seed]
            print(f"  {bname:4s} seed={seed:5d}  gamma={r['gamma']:.4f}  "
                  f"({time.time()-tt:.1f}s)")
    with open(MP_CACHE, 'wb') as f:
        pickle.dump(MP, f)
    return MP


def save_stage(name, payload):
    with open(os.path.join(RESULTS, f'v2_{name}.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=str)


def exp_A(MP):
    print("\n" + "=" * 78)
    print(" EXP A: MARGIN--DETECTION LINK DIAGNOSTIC (Table V)")
    print("=" * 78)
    bridge_rows = []
    pooled = {k: {'mmd': [], 'auc': [], 'dmin': [], 'f1': []} for k in ('q', 'rbf', 'lap')}
    for bname in ('CNC', 'CSTR', 'Tank'):
        seeds = cfg.SEEDS
        for aname in MP[bname][seeds[0]]['fam']:
            row = {'bench': bname, 'attack': ATTACK_LABELS[aname]}
            for kern in ('q', 'rbf', 'lap'):
                for met in ('mmd', 'auc', 'dmin', 'f1'):
                    vals = [MP[bname][s]['fam'][aname][kern][met] for s in seeds]
                    row[f'{kern}_{met}'] = (float(np.mean(vals)), float(np.std(vals)))
                    pooled[kern][met].extend(
                        [MP[bname][s]['fam'][aname][kern][met] for s in seeds])
            bridge_rows.append(row)

    def pool2(met):
        return (np.array(pooled['q'][met] + pooled['rbf'][met]),
                np.array(pooled['q']['f1'] + pooled['rbf']['f1']))
    corr = {}
    for met in ('mmd', 'auc', 'dmin'):
        x, y = pool2(met)
        rho, pv = spearmanr(x, y)
        corr[met] = (float(rho), float(pv))
        print(f"  Spearman rho_S({met.upper():4s}, F1) = {rho:+.3f}  (p={pv:.2e}, "
              f"n={len(x)} family-seed-kernel points)")
    per_kernel_corr = {}
    for kern in ('q', 'rbf', 'lap'):
        per_kernel_corr[kern] = {}
        for met in ('mmd', 'auc', 'dmin'):
            rho, pv = spearmanr(pooled[kern][met], pooled[kern]['f1'])
            per_kernel_corr[kern][met] = (float(rho), float(pv))

    print("\n  LaTeX rows (Table V; MMD, AUC, Dmin, F1 as Q-Dir / RBF):")
    for r in bridge_rows:
        print(f"  {r['bench']} & {r['attack']} & {r['q_mmd'][0]:+.3f} & {r['rbf_mmd'][0]:+.3f} & "
              f"{r['q_auc'][0]:.2f} & {r['rbf_auc'][0]:.2f} & "
              f"{r['q_dmin'][0]:+.2f} & {r['rbf_dmin'][0]:+.2f} & "
              f"{r['q_f1'][0]:.2f} & {r['rbf_f1'][0]:.2f} \\\\")
    print(f"  Spearman row: MMD {corr['mmd'][0]:+.2f} | AUC {corr['auc'][0]:+.2f} | "
          f"Dmin {corr['dmin'][0]:+.2f}")
    save_stage('expA', dict(rows=bridge_rows, pooled_corr=corr,
                            per_kernel_corr=per_kernel_corr))


def exp_B(MP):
    print("\n" + "=" * 78)
    print(" EXP B: TUNED BASELINES + AUTOENCODER (Table VIII new columns)")
    print("=" * 78)
    out = {}
    for bname in ('CNC', 'CSTR', 'Tank'):
        seeds = cfg.SEEDS
        for aname in MP[bname][seeds[0]]['fam']:
            oc = [MP[bname][s]['fam'][aname]['ocsvm_t'] for s in seeds]
            if_ = [MP[bname][s]['fam'][aname]['iforest_t'] for s in seeds]
            aev = [MP[bname][s]['fam'][aname]['ae'] for s in seeds]
            qf = [MP[bname][s]['fam'][aname]['q']['f1'] for s in seeds]
            out[f'{bname}_{aname}'] = dict(ocsvm_t=ms(oc), iforest_t=ms(if_), ae=ms(aev))
            print(f"{bname:5s} {ATTACK_LABELS[aname]:10s} LaTeX: {ms(oc)} & {ms(if_)} & {ms(aev)}"
                  f"   [Q-Dir ref {ms(qf)}]")
    # FPR sanity for the suspiciously uniform detectors
    fpr = {}
    for bname in ('CNC',):
        mp = MP[bname][cfg.SEEDS[0]]
        Xn_p = mp['Xn_p']
        print("  (see JSON expB_selected_params for chosen grids)")
    out['selected_params'] = {b: {str(s): MP[b][s]['tuned_params'] for s in cfg.SEEDS}
                              for b in MP}
    save_stage('expB', out)


def exp_E(MP):
    print("\n" + "=" * 78)
    print(" EXP E: SURROGATE-IMPROVEMENT ABLATION (Table VII, CNC)")
    print("=" * 78)
    abl = {v: {'eps': [], 'mae': [], 'cka': [], 'f1': []}
           for v in ('V0', 'V1', 'V2', 'V3', 'V4')}
    for seed in cfg.SEEDS:
        mp = MP['CNC'][seed]
        Xn_p, Xa_pool, qcache = mp['Xn_p'], mp['Xa_pool'], mp['qcache']
        train_X = np.vstack([Xn_p, Xa_pool[:len(Xn_p)]])
        K_train = qcache.gram(train_X)
        full_X = np.vstack([Xn_p, Xa_pool])
        K_full = qcache.gram(full_X)
        tp = np.vstack([Xn_p[:6], Xa_pool[:6]])
        K_tp = qcache.gram(tp)
        variants = {
            'V0': SurrogateVariant(15, (64, 32, 16), 500, normalized=False, seed=0).fit(train_X, K_train),
            'V1': SurrogateVariant(15, (64, 32, 16), 500, normalized=True, seed=0).fit(train_X, K_train),
            'V2': SurrogateVariant(64, (128, 64), 500, normalized=False, seed=0).fit(train_X, K_train),
            'V4': SurrogateVariant(15, (64, 32, 16), 500, normalized=True, seed=0).fit(full_X, K_full),
        }
        v3 = SurrogateVariant(15, (64, 32, 16), 500, normalized=True, seed=0).fit(train_X, K_train)
        v3.refine_head_cka(train_X, K_train, maxiter=60, seed=0)
        variants['V3'] = v3
        for vn, var in variants.items():
            f1s = []
            for aname, ainfo in mp['data']['atk'].items():
                Xa_t = (np.nan_to_num(ainfo['windows']) - mp['mu']) @ mp['V']
                f1s.append(variant_pw_f1(var, Xn_p, Xa_t))
            abl[vn]['eps'].append(var.max_error(tp, K_tp))
            abl[vn]['mae'].append(var.mae(tp, K_tp))
            abl[vn]['cka'].append(linear_cka(var.kernel(tp), K_tp))
            abl[vn]['f1'].append(np.mean(f1s))
        print(f"  seed {seed}: " + "  ".join(
            f"{vn}:F1={abl[vn]['f1'][-1]:.2f}" for vn in ('V0', 'V1', 'V2', 'V3', 'V4')))
    names = {'V0': 'MLP, $D_e{=}15$ (current)', 'V1': 'Normalized MLP',
             'V2': 'MLP, $D_e{=}64$', 'V3': 'Alignment-refined (CKA)',
             'V4': 'Expanded training pool'}
    out = {}
    for vn in ('V0', 'V1', 'V2', 'V3', 'V4'):
        row = {m: ms(abl[vn][m]) for m in ('eps', 'mae', 'cka', 'f1')}
        out[vn] = row
        print(f"  LaTeX: {names[vn]} & {row['eps']} & {row['cka']} & {row['f1']} \\\\")
    save_stage('expE', out)


def exp_C(MP, benches):
    print("\n" + "=" * 78)
    print(f" EXP C: MULTI-SEED BUDGETED SENSOR CO-DESIGN (Table IX) {benches}")
    print("=" * 78)
    from itertools import combinations
    out = {}
    for bname, budget in benches:
        p = MP[bname][cfg.SEEDS[0]]['data']['p']
        W_L = MP[bname][cfg.SEEDS[0]]['data']['W_L']
        res = {c: {'mmd': [], 'f1': []} for c in ('greedy', 'random', 'topvar', 'all')}
        sel_log = {}
        for seed in cfg.SEEDS:
            mp = MP[bname][seed]
            theta = mp['theta']
            train_X = np.vstack([mp['Xn_p'], mp['Xa_pool'][:len(mp['Xn_p'])]])
            v0 = SurrogateVariant(15, (64, 32, 16), 500, normalized=False, seed=0)
            v0.fit(train_X, mp['qcache'].gram(train_X))
            raw_atk = np.nan_to_num(np.vstack(
                [a['windows'] for a in mp['data']['atk'].values()]))
            M = separation_matrices_v2(raw_atk, v0.embed, mp['mu'], mp['V'], p, W_L)
            g_sel = greedy_logdet(M, budget)
            tv_sel = top_variance_sensors(mp['data']['nom_std'], p, W_L, budget)
            sel_log[seed] = dict(greedy=g_sel, topvar=tv_sel)
            mmd_g, f1_g = eval_mask(bname, seed, mask_from(g_sel, p), theta)
            res['greedy']['mmd'].append(mmd_g); res['greedy']['f1'].append(f1_g)
            mmd_tv, f1_tv = eval_mask(bname, seed, mask_from(tv_sel, p), theta)
            res['topvar']['mmd'].append(mmd_tv); res['topvar']['f1'].append(f1_tv)
            mmd_a, f1_a = eval_mask(bname, seed, np.ones(p), theta)
            res['all']['mmd'].append(mmd_a); res['all']['f1'].append(f1_a)
            rng = np.random.RandomState(seed + 77)
            all_masks = list(combinations(range(p), budget))
            if len(all_masks) <= 10:
                draws = all_masks
            else:
                draws = [all_masks[i] for i in rng.choice(len(all_masks), 10, replace=False)]
            r_mmd, r_f1 = [], []
            for dsel in draws:
                mm, ff = eval_mask(bname, seed, mask_from(dsel, p), theta)
                r_mmd.append(mm); r_f1.append(ff)
            res['random']['mmd'].append(np.mean(r_mmd))
            res['random']['f1'].append(np.mean(r_f1))
            print(f"  {bname} seed={seed}: greedy={g_sel} F1={f1_g:.2f} | "
                  f"topvar={tv_sel} F1={f1_tv:.2f} | rand F1={np.mean(r_f1):.2f} "
                  f"({len(draws)} draws) | all F1={f1_a:.2f}", flush=True)
        out[bname] = {'budget': budget,
                      'selected': {str(k): v for k, v in sel_log.items()},
                      **{c: dict(mmd=f"{np.mean(v['mmd']):.4f}$\\pm${np.std(v['mmd']):.4f}",
                                 f1=ms(v['f1'])) for c, v in res.items()}}
        print(f"  LaTeX ({bname} {budget}/{p}):")
        for c, lab in (('greedy', 'Greedy AS-QDT'), ('random', 'Random'),
                       ('topvar', 'Top-variance'), ('all', 'All sensors')):
            v = res[c]
            print(f"    {lab} & {np.mean(v['mmd']):.4f}$\\pm${np.std(v['mmd']):.4f} & "
                  f"{ms(v['f1'])} \\\\")
        save_stage(f'expC_{bname}', out[bname])


def exp_D(MP, benches):
    print("\n" + "=" * 78)
    print(f" EXP D: ADAPTIVE-ATTACK STRESS TEST (Table X) {benches}")
    print("=" * 78)
    for bname in benches:
        params = cfg.ATTACK_SET[bname]
        rows = {'lib': {'I': [], 'D': [], 'mmd': [], 'f1q': [], 'f1r': [], 'inset': []},
                'ada': {'I': [], 'D': [], 'mmd': [], 'f1q': [], 'f1r': [], 'inset': []}}
        for seed in cfg.SEEDS:
            mp = MP[bname][seed]
            rollout, gens, shape = ADAPTERS[bname](seed, mp)
            a_lib, lib_name, lib_ep = strongest_inset_library(bname, seed, mp, gens, shape)
            att = AdaptiveAttacker(rollout, cfg.N_EPISODES_ATK, params,
                                   mp['mu'], mp['V'], mp['qcache'], mp['S_nom'], mp['K_q'])
            ev_lib = att.evaluate(a_lib)
            f1q_l, f1r_l = pw_f1_for_fixed_attack(mp, ev_lib['Xa'])
            a_ada, ev_ada = att.optimize(a_lib, n_iters=80, seed=seed)
            f1q_a, f1r_a = pw_f1_for_fixed_attack(mp, ev_ada['Xa'])
            for tag, ev, fq, fr in (('lib', ev_lib, f1q_l, f1r_l),
                                    ('ada', ev_ada, f1q_a, f1r_a)):
                rows[tag]['I'].append(np.mean(ev['impacts']))
                rows[tag]['D'].append(np.mean(ev['legacies']))
                rows[tag]['mmd'].append(ev['mmd'])
                rows[tag]['f1q'].append(fq)
                rows[tag]['f1r'].append(fr)
                rows[tag]['inset'].append(ev['in_set'])
            print(f"  {bname} seed={seed} [init={lib_name}/ep{lib_ep}] "
                  f"lib: mmd={ev_lib['mmd']:+.4f} F1q={f1q_l:.2f} in={ev_lib['in_set']}/10 | "
                  f"ada: mmd={ev_ada['mmd']:+.4f} F1q={f1q_a:.2f} F1r={f1r_a:.2f} "
                  f"in={ev_ada['in_set']}/10", flush=True)
        out = {tag: dict(I=f"{np.mean(v['I']):.2f}", D=f"{np.mean(v['D']):.1f}",
                         mmd=f"{np.mean(v['mmd']):+.4f}$\\pm${np.std(v['mmd']):.4f}",
                         f1q=ms(v['f1q']), f1r=ms(v['f1r']),
                         inset=f"{np.mean(v['inset']):.1f}/10")
               for tag, v in rows.items()}
        for tag, lab in (('lib', 'Library'), ('ada', 'Adaptive')):
            v = rows[tag]
            print(f"  LaTeX: {bname} & {lab} & {np.mean(v['I']):.2f} & "
                  f"{np.mean(v['D']):.1f} & {np.mean(v['mmd']):+.4f} & "
                  f"{ms(v['f1q'])} & {ms(v['f1r'])} \\\\   "
                  f"(in-set {np.mean(v['inset']):.1f}/10)")
        save_stage(f'expD_{bname}', out)


def main():
    t0 = time.time()
    stages = sys.argv[1:] or ['A', 'B', 'E', 'C_CNC', 'C_CSTR', 'D_CNC', 'D_CSTR', 'D_Tank']
    sanity_check_batch_statevectors()
    MP = load_or_build_mp(stages)
    if 'A' in stages: exp_A(MP)
    if 'B' in stages: exp_B(MP)
    if 'E' in stages: exp_E(MP)
    if 'C_CNC' in stages: exp_C(MP, [('CNC', 4)])
    if 'C_CSTR' in stages: exp_C(MP, [('CSTR', 2)])
    if 'D_CNC' in stages: exp_D(MP, ['CNC'])
    if 'D_CSTR' in stages: exp_D(MP, ['CSTR'])
    if 'D_Tank' in stages: exp_D(MP, ['Tank'])
    print(f"\nSTAGE(S) {stages} DONE: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
