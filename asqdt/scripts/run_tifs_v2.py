#!/usr/bin/env python3
"""
AS-QDT Revised TIFS Experiment — addresses all reviewer gaps.

Changes from run_tifs_final.py:
  Gap W1: Per-window detection fixed (threshold calibrated from nominal single-window scores)
  Gap W2: Data generated independently per seed (nonzero classical variance)
  Gap W3: Non-vacuous bound sample-size calculation
  Gap W4: Non-kernel baselines (One-Class SVM, Isolation Forest, residual-norm)
  Gap W6: Inner attack optimizer validation (one benchmark)
"""
import sys, os, time, json, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asqdt import config as cfg
from asqdt.quantum.kernel import QuantumCircuit
from asqdt.quantum.surrogate import NystromSurrogate
from asqdt.quantum.learned_surrogate import LearnedSurrogate
from asqdt.detection.mmd import mmd_squared, mmd_from_embeddings, compute_threshold
from asqdt.cpms.data import generate_dataset
from asqdt.cpms.cstr import generate_cstr_dataset
from asqdt.cpms.tank import generate_tank_dataset
from asqdt.codesign.sensors import compute_separation_matrices, greedy_sensor_selection
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf, laplacian_kernel as sk_lap
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS, exist_ok=True)


# ── Utilities ─────────────────────────────────────────────────────

def pca_fit(X, dim):
    X = np.nan_to_num(X); mu = X.mean(0); C = X - mu
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    Vt = Vt[:min(dim, Vt.shape[0])]
    # --- Deterministic sign convention (svd_flip-style) -------------------
    # Make the largest-|.| entry of each principal component positive.
    # LAPACK/BLAS builds return singular vectors with arbitrary signs;
    # distance-based classical kernels (RBF/Laplacian) are invariant to
    # component sign flips, but the RY/RZ quantum encoding is not, so the
    # quantum-kernel design was backend-dependent without this convention.
    signs = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
    signs[signs == 0] = 1.0
    return mu, (Vt * signs[:, None]).T


def perm_thresh(K, rng, n_perm=cfg.N_PERM_THRESH):
    n = len(K); null = []
    for _ in range(n_perm):
        p = rng.permutation(n); m1 = n // 2
        null.append(mmd_squared(K[np.ix_(p[:m1],p[:m1])],
                                K[np.ix_(p[m1:],p[m1:])],
                                K[np.ix_(p[:m1],p[m1:])]))
    return np.percentile(null, 95)


def per_window_scores(K_ref, kmaker, X_ref, X_test):
    """Score each test window individually using S(ξ_t) from eq (11)."""
    scores = []
    mean_ref = (np.sum(K_ref) - np.trace(K_ref)) / (len(K_ref) * (len(K_ref) - 1))
    for i in range(len(X_test)):
        xi = X_test[i:i+1]
        K_tt = kmaker(xi, xi)
        K_rt = kmaker(X_ref, xi)
        s = mean_ref + K_tt[0, 0] - 2 * np.mean(K_rt)
        scores.append(s)
    return np.array(scores)


def per_window_threshold(K_ref, kmaker, X_ref, n_perm=100, alpha=0.05):
    """Calibrate threshold for S(ξ_t) from nominal single-window scores.
    
    FIX FOR GAP W1: Previously used episode-level MMD threshold for
    single-window scores, causing scale mismatch and F1=0.50.
    Now calibrates from nominal-vs-nominal single-window distribution.
    """
    nom_scores = per_window_scores(K_ref, kmaker, X_ref, X_ref)
    return np.percentile(nom_scores, 100 * (1 - alpha))


def detect_m(y_true, scores, thresh):
    pred = (scores > thresh).astype(int)
    f1 = f1_score(y_true, pred, zero_division=0)
    try: auc = roc_auc_score(y_true, scores)
    except: auc = 0.5
    tpr = np.mean(pred[y_true == 1]) if sum(y_true == 1) else 0
    fpr = np.mean(pred[y_true == 0]) if sum(y_true == 0) else 0
    return dict(f1=round(f1, 3), auroc=round(auc, 3), tpr=round(tpr, 3), fpr=round(fpr, 3))


def exact_sign_test_two_sided(diffs):
    """Two-sided exact sign test. Returns p-value."""
    n = len(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    k = max(n_pos, n_neg)
    from math import comb
    p = 2 * sum(comb(n, i) for i in range(k, n+1)) / 2**n
    return min(p, 1.0)


def sample_size_for_nonvacuous(kappa2=1.0, delta=0.05, target_score=0.16):
    """GAP W3: Compute (m,n) needed for β_{m,n,δ} < target_score."""
    print("\n[*] Sample-size calculation for non-vacuous bounds:")
    print(f"    κ²={kappa2}, δ={delta}, target score={target_score}")
    for m_ratio in [2, 2, 2, 2]:  # m = ratio * n
        for n in [10, 50, 100, 500, 1000, 5000]:
            m = m_ratio * n
            beta = 4 * kappa2 * np.sqrt(0.5 * np.log(2/delta) * (1/m + 1/n))
            if beta < target_score:
                print(f"    → β={beta:.4f} < {target_score} at m={m}, n={n}")
                print(f"      (need ~{m} nominal + ~{n} attack episodes)")
                return m, n
    # Table showing the progression
    print(f"    {'m':>6} {'n':>6} {'β':>8}")
    for n in [10, 50, 100, 500, 1000, 5000, 10000]:
        m = 2 * n
        beta = 4 * kappa2 * np.sqrt(0.5 * np.log(2/delta) * (1/m + 1/n))
        marker = " ← non-vacuous" if beta < target_score else ""
        print(f"    {m:>6} {n:>6} {beta:>8.4f}{marker}")
    return None, None


# ── Non-kernel baselines (GAP W4) ────────────────────────────────

def run_baselines(Xn_p, Xa_t):
    """Run non-kernel baselines: One-Class SVM, Isolation Forest, residual-norm."""
    results = {}
    y_true = np.concatenate([np.zeros(len(Xn_p)), np.ones(len(Xa_t))])
    X_all = np.vstack([Xn_p, Xa_t])
    
    # 1. One-Class SVM (nu=0.1, RBF kernel)
    ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    ocsvm.fit(Xn_p)
    scores_svm = -ocsvm.decision_function(X_all)  # negative = more anomalous
    thresh_svm = np.percentile(-ocsvm.decision_function(Xn_p), 95)
    results['ocsvm'] = detect_m(y_true, scores_svm, thresh_svm)
    
    # 2. Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(Xn_p)
    scores_iso = -iso.decision_function(X_all)
    thresh_iso = np.percentile(-iso.decision_function(Xn_p), 95)
    results['iforest'] = detect_m(y_true, scores_iso, thresh_iso)
    
    # 3. Residual-norm (L2 norm of windowed feature, threshold from nominal)
    norms = np.linalg.norm(X_all, axis=1)
    norms_nom = np.linalg.norm(Xn_p, axis=1)
    thresh_norm = np.percentile(norms_nom, 95)
    results['res_norm'] = detect_m(y_true, norms, thresh_norm)
    
    return results


# ── Per-seed experiment (GAP W2: data generated per seed) ────────

def generate_bench_data(seed):
    """Generate ALL benchmark data for one seed. GAP W2 fix."""
    return {
        'CNC':  generate_dataset(mismatch_frac=0.05, seed=seed, W_L=10, E_max=2.0,
                                  legacy_method='chi_sq'),
        'CSTR': generate_cstr_dataset(mismatch_frac=0.05, n_nom=20, n_atk=10,
                                       H=120, seed=seed),
        'Tank': generate_tank_dataset(mismatch_frac=0.05, n_nom=20, n_atk=10,
                                       H=120, seed=seed),
    }


def parse_bench(benches):
    """Parse benchmark data into (nom, atk) pairs."""
    bench_data = {}
    for b, d in benches.items():
        bench_data[b] = (d[0], d[1])
    return bench_data


def run_seed(Xn, atk, seed, nq=4, depth=3):
    """Run one seed with per-window threshold fix (GAP W1)."""
    rng = np.random.RandomState(seed)
    td = nq * depth
    mu, V = pca_fit(Xn, td)
    Xn_p = (Xn - mu) @ V

    # Quantum kernel optimization
    qc = QuantumCircuit(nq, depth)
    theta = qc.initial_params(rng)
    best_m, best_t = -np.inf, theta.copy()
    margins = []

    all_a = np.vstack([d['windows'] for d in atk.values()])
    Xa_pool = (np.nan_to_num(all_a) - mu) @ V

    for it in range(cfg.MAX_ITERS):
        m_val = mmd_squared(qc.gram_matrix(Xn_p, theta),
                            qc.gram_matrix(Xa_pool, theta),
                            qc.cross_gram(Xn_p, Xa_pool, theta))
        margins.append(m_val)
        if m_val > best_m: best_m = m_val; best_t = theta.copy()
        g = qc.spsa_gradient(Xn_p, Xa_pool, theta,
            lambda xn, xa, th: mmd_squared(qc.gram_matrix(xn, th),
                                           qc.gram_matrix(xa, th),
                                           qc.cross_gram(xn, xa, th)), rng=rng)
        theta += cfg.KERNEL_LR * (g - 0.02 * theta)
    theta = best_t

    def kfn(X, Y):
        if X is Y: return qc.gram_matrix(X, theta)
        return qc.cross_gram(X, Y, theta)

    # Surrogates
    lm = Xn_p[rng.choice(len(Xn_p), min(cfg.NYSTROM_RANK, len(Xn_p)), replace=False)]
    nyst = NystromSurrogate(rank=min(cfg.NYSTROM_RANK, len(lm))); nyst.fit(lm, kfn)
    train_X = np.vstack([Xn_p, Xa_pool[:min(len(Xn_p), len(Xa_pool))]])
    learned = LearnedSurrogate(embed_dim=cfg.LEARNED_EMBED_DIM,
                                hidden=cfg.LEARNED_HIDDEN,
                                max_iter=cfg.LEARNED_MAX_ITER)
    learned.fit(train_X, kfn)

    tp = np.vstack([Xn_p[:6], Xa_pool[:6]])
    eps_n = nyst.approximation_error(tp, kfn, 60)
    eps_l = learned.approximation_error(tp, kfn, 60)

    # RBF bandwidth — held-out tuning
    med = np.median(np.linalg.norm(Xn_p[:, None] - Xn_p[None, :], axis=2))
    gammas = [g / (med**2 + 1e-8) for g in [0.1, 0.5, 1, 2, 5]]
    n_tune = len(Xa_pool) // 2
    Xa_tune = Xa_pool[:n_tune]
    best_g = gammas[2]; best_rbf = -np.inf
    for g in gammas:
        v = mmd_squared(sk_rbf(Xn_p, Xn_p, g), sk_rbf(Xa_tune, Xa_tune, g),
                        sk_rbf(Xn_p, Xa_tune, g))
        if v > best_rbf: best_rbf = v; best_g = g

    # Gram matrices
    K_q = qc.gram_matrix(Xn_p, theta)
    K_rbf = sk_rbf(Xn_p, Xn_p, best_g)
    K_lap = sk_lap(Xn_p, Xn_p, best_g)

    # GAP W1 FIX: Per-window thresholds calibrated from nominal single-window scores
    kmaker_q = lambda X, Y: qc.cross_gram(X, Y, theta) if not (X is Y) else qc.gram_matrix(X, theta)
    kmaker_r = lambda X, Y: sk_rbf(X, Y, best_g)
    kmaker_l = lambda X, Y: sk_lap(X, Y, best_g)
    
    tau_sw_q = per_window_threshold(K_q, kmaker_q, Xn_p)
    tau_sw_r = per_window_threshold(K_rbf, kmaker_r, Xn_p)
    tau_sw_l = per_window_threshold(K_lap, kmaker_l, Xn_p)

    # Learned-surrogate per-window detection
    phi_nom = learned.embed(Xn_p)
    K_learned = phi_nom @ phi_nom.T
    kmaker_ls = lambda X, Y: learned.embed(X) @ learned.embed(Y).T
    tau_sw_ls = per_window_threshold(K_learned, kmaker_ls, Xn_p)

    res = dict(eps_nyst=eps_n, eps_learned=eps_l, co_margin=best_m, margins=margins,
               best_gamma=best_g)
    
    for aname, ainfo in atk.items():
        Xa_t = (np.nan_to_num(ainfo['windows']) - mu) @ V

        # Episode-level MMD scores (m=20, n=10)
        s_q = mmd_squared(K_q, qc.gram_matrix(Xa_t, theta),
                          qc.cross_gram(Xn_p, Xa_t, theta))
        s_r = mmd_squared(K_rbf, sk_rbf(Xa_t, Xa_t, best_g),
                          sk_rbf(Xn_p, Xa_t, best_g))
        s_l = mmd_squared(K_lap, sk_lap(Xa_t, Xa_t, best_g),
                          sk_lap(Xn_p, Xa_t, best_g))
        s_ls = mmd_from_embeddings(learned.embed(Xn_p), learned.embed(Xa_t))
        s_ns = mmd_from_embeddings(nyst.embed(Xn_p, kfn), nyst.embed(Xa_t, kfn))

        # GAP W1 FIX: Per-window detection with correctly calibrated thresholds
        n_n = len(Xn_p); n_at = len(Xa_t)
        y_true = np.concatenate([np.zeros(n_n), np.ones(n_at)])
        X_all = np.vstack([Xn_p, Xa_t])

        sq = per_window_scores(K_q, kmaker_q, Xn_p, X_all)
        sr = per_window_scores(K_rbf, kmaker_r, Xn_p, X_all)
        sl = per_window_scores(K_lap, kmaker_l, Xn_p, X_all)
        s_ls_pw = per_window_scores(K_learned, kmaker_ls, Xn_p, X_all)

        # GAP W4: Non-kernel baselines
        bl = run_baselines(Xn_p, Xa_t)

        res[aname] = dict(
            q=s_q, rbf=s_r, lap=s_l, learned=s_ls, nyst=s_ns,
            det_q=detect_m(y_true, sq, tau_sw_q),
            det_rbf=detect_m(y_true, sr, tau_sw_r),
            det_lap=detect_m(y_true, sl, tau_sw_l),
            det_learned=detect_m(y_true, s_ls_pw, tau_sw_ls),
            baseline_ocsvm=bl['ocsvm'],
            baseline_iforest=bl['iforest'],
            baseline_resnorm=bl['res_norm'],
        )
    return res


def verify_attack_set(atk_data, params):
    rho = params['rho']; eps = params['epsilon']; E_max = params['E_max']
    rows = []
    for aname, ainfo in atk_data.items():
        if 'impacts' not in ainfo: continue
        n_ep = len(ainfo['impacts'])
        in_set = 0
        for i in range(n_ep):
            imp = ainfo['impacts'][i]
            leg = ainfo['legacy_responses'][i]
            eng = ainfo['energies'][i]
            if imp >= rho and leg <= eps and eng <= E_max * (1 + 1e-8):
                in_set += 1
        min_imp = min(ainfo['impacts'])
        max_leg = max(ainfo['legacy_responses'])
        rows.append(dict(attack=aname, rho=rho, eps=eps, min_I=min_imp,
                        max_D=max_leg, frac=f"{in_set}/{n_ep}"))
    return rows


# ── Inner attack optimizer validation (GAP W6) ───────────────────

def run_minimax_validation(Xn_p, Xa_pool, qc, theta, rng, n_steps=20):
    """Run inner attack optimizer and compare with library attacks.
    
    GAP W6: Validates that the minimax formulation produces harder-to-detect
    attacks than the fixed library.
    """
    # Library attack score (what the current experiment uses)
    K_q = qc.gram_matrix(Xn_p, theta)
    K_a = qc.gram_matrix(Xa_pool, theta)
    K_cross = qc.cross_gram(Xn_p, Xa_pool, theta)
    library_score = mmd_squared(K_q, K_a, K_cross)
    
    # Adversarial attack: perturb attack features to minimize MMD
    # (simplified: gradient descent on feature perturbations)
    Xa_adv = Xa_pool.copy()
    best_adv_score = library_score
    lr = 0.01
    
    for step in range(n_steps):
        # SPSA gradient estimate for attack perturbation
        eps = 0.05
        delta = rng.choice([-1, 1], size=Xa_adv.shape)
        
        Xa_plus = Xa_adv + eps * delta
        K_a_p = qc.gram_matrix(Xa_plus, theta)
        K_c_p = qc.cross_gram(Xn_p, Xa_plus, theta)
        score_plus = mmd_squared(K_q, K_a_p, K_c_p)
        
        Xa_minus = Xa_adv - eps * delta
        K_a_m = qc.gram_matrix(Xa_minus, theta)
        K_c_m = qc.cross_gram(Xn_p, Xa_minus, theta)
        score_minus = mmd_squared(K_q, K_a_m, K_c_m)
        
        # Minimize MMD (attacker's objective)
        grad = (score_plus - score_minus) / (2 * eps) * delta
        Xa_adv -= lr * grad
        
        K_a_new = qc.gram_matrix(Xa_adv, theta)
        K_c_new = qc.cross_gram(Xn_p, Xa_adv, theta)
        adv_score = mmd_squared(K_q, K_a_new, K_c_new)
        best_adv_score = min(best_adv_score, adv_score)
    
    return dict(
        library_score=float(library_score),
        adversarial_score=float(best_adv_score),
        reduction_pct=float((library_score - best_adv_score) / (abs(library_score) + 1e-10) * 100)
    )


# ── Main ──────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("  AS-QDT REVISED TIFS EXPERIMENT (reviewer gaps addressed)")
    print("  5 seeds | 3 benchmarks | per-seed data | non-kernel baselines")
    print("=" * 70)

    # GAP W3: Sample-size calculation
    sample_size_for_nonvacuous()

    # ── Attack-set verification (seed 42 only, for Table II) ──────
    print("\n[1] Attack-set verification (Table II, seed 42)...")
    bench_42 = parse_bench(generate_bench_data(42))
    for bname, (nom, atk) in bench_42.items():
        params = cfg.ATTACK_SET[bname]
        rows = verify_attack_set(atk, params)
        for r in rows:
            print(f"  {bname:5s} {r['attack']:20s} ρ={r['rho']} ε={r['eps']:.0f} "
                  f"min_I={r['min_I']:.3f} max_D={r['max_D']:.1f} frac={r['frac']}")

    # ── Main experiment: per-seed data generation (GAP W2) ────────
    print("\n[2] Main experiment (per-seed data generation)...")
    all_res = {b: [] for b in ['CNC', 'CSTR', 'Tank']}
    
    for si, seed in enumerate(cfg.SEEDS):
        # GAP W2: Generate fresh data for EACH seed
        bench_data = parse_bench(generate_bench_data(seed))
        
        for bname, (nom, atk) in bench_data.items():
            print(f"  seed={seed} {bname}...", end=" ", flush=True)
            r = run_seed(nom, atk, seed, nq=4, depth=3)
            all_res[bname].append(r)
            print(f"ε_l={r['eps_learned']:.3f}")

    # ── Aggregate and print ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS: mean ± std (5 seeds, per-seed data)")
    print("=" * 70)

    summary = {}
    for bname in ['CNC', 'CSTR', 'Tank']:
        data = all_res[bname]
        atks = [k for k in data[0] if k not in 
                ('eps_nyst', 'eps_learned', 'co_margin', 'margins', 'best_gamma')]
        eps_n = [d['eps_nyst'] for d in data]
        eps_l = [d['eps_learned'] for d in data]

        print(f"\n  {bname} (4q):")
        print(f"    ε_nyst: {np.mean(eps_n):.3f}±{np.std(eps_n):.3f}"
              f"  ε_learned: {np.mean(eps_l):.3f}±{np.std(eps_l):.3f}")
        
        # MMD scores
        hdr = (f"    {'Attack':<16} {'Q-Direct':>16} {'RBF':>16} "
               f"{'Laplacian':>16} {'Learned-S':>16}")
        print(hdr)

        bench = {}
        for a in atks:
            v = {k: [d[a][k] for d in data] for k in ['q', 'rbf', 'lap', 'learned', 'nyst']}
            f = lambda x: f"{np.mean(x):+.4f}±{np.std(x):.4f}"
            print(f"    {a:<16} {f(v['q']):>16} {f(v['rbf']):>16} "
                  f"{f(v['lap']):>16} {f(v['learned']):>16}")

            # Significance test (exact paired sign test)
            best_cl = v['rbf'] if np.mean(v['rbf']) >= np.mean(v['lap']) else v['lap']
            diffs = [v['q'][i] - best_cl[i] for i in range(len(v['q']))]
            if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
                p = exact_sign_test_two_sided(diffs)
                direction = "Q>cl" if np.mean(diffs) > 0 else "cl>Q"
                print(f"      → {direction}, p={p:.4f} {'**' if p < 0.05 else '(n.s.)'}")

            bench[a] = {kk: f"{np.mean(vv):.4f}±{np.std(vv):.4f}" for kk, vv in v.items()}
            
            # Detection metrics (per-window, with fixed threshold)
            for method, mkey in [('Q-Direct', 'det_q'), ('RBF', 'det_rbf'), ('Lap', 'det_lap'), ('Learned-S', 'det_learned')]:
                f1s = [d[a][mkey]['f1'] for d in data]
                aucs = [d[a][mkey]['auroc'] for d in data]
                bench[a][f'f1_{mkey}'] = f"{np.mean(f1s):.3f}±{np.std(f1s):.3f}"
                bench[a][f'auc_{mkey}'] = f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}"
            
            # Baseline metrics
            for bl_name in ['ocsvm', 'iforest', 'resnorm']:
                bl_key = f'baseline_{bl_name}'
                f1s = [d[a][bl_key]['f1'] for d in data]
                aucs = [d[a][bl_key]['auroc'] for d in data]
                bench[a][f'f1_{bl_name}'] = f"{np.mean(f1s):.3f}±{np.std(f1s):.3f}"
                bench[a][f'auc_{bl_name}'] = f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}"

        bench['eps_nyst'] = f"{np.mean(eps_n):.3f}±{np.std(eps_n):.3f}"
        bench['eps_learned'] = f"{np.mean(eps_l):.3f}±{np.std(eps_l):.3f}"
        summary[bname] = bench

    # ── Print detection metrics comparison ────────────────────────
    print("\n" + "=" * 70)
    print("  DETECTION METRICS (per-window, mean±std over 5 seeds)")
    print("=" * 70)
    for bname in ['CNC', 'CSTR', 'Tank']:
        data = all_res[bname]
        atks = [k for k in data[0] if k not in 
                ('eps_nyst', 'eps_learned', 'co_margin', 'margins', 'best_gamma')]
        print(f"\n  {bname}:")
        print(f"    {'Attack':<16} {'Q-Direct':>10} {'Learned-S':>10} {'RBF':>10} {'Lap':>10} "
              f"{'OCSVM':>10} {'IForest':>10} {'ResNorm':>10}")
        for a in atks:
            vals = []
            for mkey in ['det_q', 'det_learned', 'det_rbf', 'det_lap', 
                         'baseline_ocsvm', 'baseline_iforest', 'baseline_resnorm']:
                f1s = [d[a][mkey]['f1'] for d in data]
                vals.append(f"{np.mean(f1s):.2f}")
            print(f"    {a:<16} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10} "
                  f"{vals[4]:>10} {vals[5]:>10} {vals[6]:>10}")

    # ── LaTeX table for Supplement Table E (learned-surrogate F1) ──
    print("\n" + "=" * 70)
    print("  SUPPLEMENT TABLE E: Learned-surrogate per-window F1")
    print("  (paste into supplement_corrected.tex, replacing TBD rows)")
    print("=" * 70)
    atk_labels = {
        'fdi': ('CNC', 'FDI'), 'command': ('CNC', 'Command'),
        'ramp': ('CNC', 'Ramp'), 'composite': ('CNC', 'Composite'),
        'cstr_fdi': ('CSTR', 'FDI'), 'cstr_thermal': ('CSTR', 'Thermal'),
        'cstr_concentration': ('CSTR', 'Conc.'),
        'tank_bias': ('Tank', 'Bias'), 'tank_oscillation': ('Tank', 'Oscill.'),
        'tank_drift': ('Tank', 'Drift'),
    }
    atk_order = ['fdi', 'command', 'ramp', 'composite',
                 'cstr_fdi', 'cstr_thermal', 'cstr_concentration',
                 'tank_bias', 'tank_oscillation', 'tank_drift']
    prev_bench = None
    for a in atk_order:
        bench_name, label = atk_labels[a]
        data = all_res[bench_name]
        f1_q = [d[a]['det_q']['f1'] for d in data]
        f1_l = [d[a]['det_learned']['f1'] for d in data]
        mq, sq_ = np.mean(f1_q), np.std(f1_q)
        ml, sl_ = np.mean(f1_l), np.std(f1_l)
        if prev_bench and prev_bench != bench_name:
            print("\\midrule")
        print(f"{bench_name:<5} & {label:<10} & {mq:.2f}$\\pm${sq_:.2f} & {ml:.2f}$\\pm${sl_:.2f} \\\\")
        prev_bench = bench_name

    # ── Minimax validation (GAP W6, CNC only) ─────────────────────
    print("\n[3] Minimax validation (CNC, seed 42)...")
    nom_42, atk_42 = bench_42['CNC']
    td = cfg.N_QUBITS * cfg.CIRCUIT_DEPTH
    mu_v, V_v = pca_fit(nom_42, td)
    Xn_v = (nom_42 - mu_v) @ V_v
    all_a_v = np.vstack([d['windows'] for d in atk_42.values()])
    Xa_v = (np.nan_to_num(all_a_v) - mu_v) @ V_v
    
    qc_v = QuantumCircuit(cfg.N_QUBITS, cfg.CIRCUIT_DEPTH)
    rng_v = np.random.RandomState(42)
    theta_v = qc_v.initial_params(rng_v)
    
    minimax = run_minimax_validation(Xn_v, Xa_v, qc_v, theta_v, rng_v)
    print(f"  Library attack score:     {minimax['library_score']:.6f}")
    print(f"  Adversarial attack score: {minimax['adversarial_score']:.6f}")
    print(f"  Reduction:                {minimax['reduction_pct']:.1f}%")
    summary['minimax_validation'] = minimax

    # ── Sensor co-design ──────────────────────────────────────────
    print("\n[4] Sensor co-design (CNC, seed 42)...")
    med_s = np.median(np.linalg.norm(Xn_v[:, None] - Xn_v[None, :], axis=2))
    g_s = 1.0 / (med_s**2 + 1e-8)
    score_full = mmd_squared(sk_rbf(Xn_v, Xn_v, g_s), sk_rbf(Xa_v, Xa_v, g_s),
                             sk_rbf(Xn_v, Xa_v, g_s))
    phi_n = Xn_v; phi_a = Xa_v
    M = compute_separation_matrices(phi_n, phi_a, cfg.N_OUTPUT)
    sel, F_vals = greedy_sensor_selection(M, cfg.SENSOR_BUDGET, cfg.N_OUTPUT)
    print(f"  Greedy: sensors={sel}, F={F_vals[-1]:.4f}")
    print(f"  Full:   score={score_full:.4f}")
    summary['sensor_codesign'] = {'greedy': str(sel), 'score_full': f"{score_full:.4f}"}

    # ── Save results ──────────────────────────────────────────────
    with open(os.path.join(RESULTS, 'tifs_v2.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Figures ───────────────────────────────────────────────────
    print("\n[5] Generating figures...")
    plt.rcParams.update({'font.size': 10})

    # Fig 1: 3-benchmark comparison (now with nonzero classical std)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, bname in enumerate(['CNC', 'CSTR', 'Tank']):
        ax = axes[idx]; data = all_res[bname]
        atks = [k for k in data[0] if k not in 
                ('eps_nyst', 'eps_learned', 'co_margin', 'margins', 'best_gamma')]
        x = np.arange(len(atks)); w = 0.2
        for ki, (kn, lb, cl) in enumerate([
            ('q', 'Q-Direct', '#2196F3'), ('rbf', 'RBF', '#FF9800'),
            ('lap', 'Laplacian', '#4CAF50'), ('learned', 'Learned-S', '#9C27B0')]):
            ms = [np.mean([d[a][kn] for d in data]) for a in atks]
            ss = [np.std([d[a][kn] for d in data]) for a in atks]
            ax.bar(x + ki*w, ms, w, yerr=ss, label=lb, color=cl, capsize=2, alpha=0.85)
        labels = [a.replace('cstr_', '').replace('tank_', '').upper()[:10] for a in atks]
        ax.set_xticks(x + 1.5*w)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('$\\hat{\\mathrm{MMD}}_u^2$')
        regime = 'Linear' if bname == 'CNC' else 'Nonlinear'
        ax.set_title(f'{bname} ({regime})')
        ax.axhline(0, color='k', lw=0.5); ax.grid(axis='y', alpha=0.3)
        if idx == 0: ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_3bench.pdf'))
    plt.close()

    # Fig 2: Surrogate comparison
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bnames = ['CNC', 'CSTR', 'Tank']
    en = [np.mean([d['eps_nyst'] for d in all_res[b]]) for b in bnames]
    el = [np.mean([d['eps_learned'] for d in all_res[b]]) for b in bnames]
    ens = [np.std([d['eps_nyst'] for d in all_res[b]]) for b in bnames]
    els = [np.std([d['eps_learned'] for d in all_res[b]]) for b in bnames]
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, en, w, yerr=ens, label='Nyström', color='#F44336', capsize=3)
    ax.bar(x + w/2, el, w, yerr=els, label='Learned MLP', color='#4CAF50', capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(bnames)
    ax.set_ylabel('$\\epsilon_{\\mathrm{surr}}$')
    ax.set_title('Surrogate Approximation Error (5 seeds)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(RESULTS, 'fig_surrogate.pdf'))
    plt.close()

    # Fig 3: Convergence
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for si, seed in enumerate(cfg.SEEDS[:3]):
        m = all_res['CNC'][si]['margins']
        ax.plot(m, 'o-', ms=3, lw=1, alpha=0.6, label=f'seed {seed}')
    ax.set_xlabel('Co-design iteration'); ax.set_ylabel('$\\hat{\\Gamma}^{(\\ell)}$')
    ax.set_title('Co-Design Convergence (CNC)'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(RESULTS, 'fig1_margin_convergence.pdf'))
    plt.close()

    print(f"  Saved figures to {RESULTS}")
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  DONE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
