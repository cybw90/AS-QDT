#!/usr/bin/env python3
"""
Compute learned-surrogate per-window detection F1 for Supplementary Table E.

Usage (from the asqdt/ project root):
    python -m asqdt.scripts.surrogate_detect

Or standalone:
    cd /path/to/asqdt
    python asqdt/scripts/surrogate_detect.py

Produces LaTeX table rows ready to paste into supplement_corrected.tex.
Runtime: ~2-4 min on Apple M2 (same 5 seeds × 3 benchmarks).
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asqdt import config as cfg
from asqdt.quantum.kernel import QuantumCircuit
from asqdt.quantum.learned_surrogate import LearnedSurrogate
from asqdt.detection.mmd import mmd_squared
from asqdt.cpms.data import generate_dataset
from asqdt.cpms.cstr import generate_cstr_dataset
from asqdt.cpms.tank import generate_tank_dataset
from sklearn.metrics import f1_score, roc_auc_score


# ── Utilities (copied from run_tifs_v2 for self-containment) ──────

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


def per_window_scores(K_ref, kmaker, X_ref, X_test):
    """Score each test window individually using S(xi_t) from eq (7)."""
    mean_ref = (np.sum(K_ref) - np.trace(K_ref)) / (len(K_ref) * (len(K_ref) - 1))
    scores = []
    for i in range(len(X_test)):
        xi = X_test[i:i+1]
        K_tt = kmaker(xi, xi)
        K_rt = kmaker(X_ref, xi)
        s = mean_ref + K_tt[0, 0] - 2 * np.mean(K_rt)
        scores.append(s)
    return np.array(scores)


def per_window_threshold(K_ref, kmaker, X_ref, alpha=0.05):
    """Calibrate threshold from nominal single-window scores."""
    nom_scores = per_window_scores(K_ref, kmaker, X_ref, X_ref)
    return np.percentile(nom_scores, 100 * (1 - alpha))


def detect_metrics(y_true, scores, thresh):
    pred = (scores > thresh).astype(int)
    f1 = f1_score(y_true, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    return dict(f1=round(f1, 3), auroc=round(auc, 3))


# ── Data generation ───────────────────────────────────────────────

def generate_bench_data(seed):
    return {
        'CNC':  generate_dataset(mismatch_frac=0.05, seed=seed, W_L=10, E_max=2.0,
                                  legacy_method='chi_sq'),
        'CSTR': generate_cstr_dataset(mismatch_frac=0.05, n_nom=20, n_atk=10,
                                       H=120, seed=seed),
        'Tank': generate_tank_dataset(mismatch_frac=0.05, n_nom=20, n_atk=10,
                                       H=120, seed=seed),
    }


# ── Per-seed computation ─────────────────────────────────────────

def run_seed_surrogate_detect(Xn, atk, seed, nq=4, depth=3):
    """Run one seed: train quantum kernel + learned surrogate, compute per-window F1."""
    rng = np.random.RandomState(seed)
    td = nq * depth

    # PCA
    mu, V = pca_fit(Xn, td)
    Xn_p = (Xn - mu) @ V

    # Quantum kernel optimization (same as run_tifs_v2)
    qc = QuantumCircuit(nq, depth)
    theta = qc.initial_params(rng)
    best_m, best_t = -np.inf, theta.copy()

    all_a = np.vstack([d['windows'] for d in atk.values()])
    Xa_pool = (np.nan_to_num(all_a) - mu) @ V

    for it in range(cfg.MAX_ITERS):
        m_val = mmd_squared(qc.gram_matrix(Xn_p, theta),
                            qc.gram_matrix(Xa_pool, theta),
                            qc.cross_gram(Xn_p, Xa_pool, theta))
        if m_val > best_m:
            best_m = m_val; best_t = theta.copy()
        g = qc.spsa_gradient(Xn_p, Xa_pool, theta,
            lambda xn, xa, th: mmd_squared(qc.gram_matrix(xn, th),
                                           qc.gram_matrix(xa, th),
                                           qc.cross_gram(xn, xa, th)), rng=rng)
        theta += cfg.KERNEL_LR * (g - 0.02 * theta)
    theta = best_t

    def kfn(X, Y):
        if X is Y:
            return qc.gram_matrix(X, theta)
        return qc.cross_gram(X, Y, theta)

    # Train learned surrogate
    train_X = np.vstack([Xn_p, Xa_pool[:min(len(Xn_p), len(Xa_pool))]])
    learned = LearnedSurrogate(embed_dim=cfg.LEARNED_EMBED_DIM,
                                hidden=cfg.LEARNED_HIDDEN,
                                max_iter=cfg.LEARNED_MAX_ITER)
    learned.fit(train_X, kfn)

    # Surrogate approximation error
    tp = np.vstack([Xn_p[:6], Xa_pool[:6]])
    eps_l = learned.approximation_error(tp, kfn, 60)

    # Q-Direct kernel matrices and threshold
    K_q = qc.gram_matrix(Xn_p, theta)
    kmaker_q = lambda X, Y: qc.cross_gram(X, Y, theta) if not (X is Y) else qc.gram_matrix(X, theta)
    tau_sw_q = per_window_threshold(K_q, kmaker_q, Xn_p)

    # Learned surrogate kernel matrices and threshold
    phi_nom = learned.embed(Xn_p)
    K_learned = phi_nom @ phi_nom.T
    kmaker_ls = lambda X, Y: learned.embed(X) @ learned.embed(Y).T
    tau_sw_ls = per_window_threshold(K_learned, kmaker_ls, Xn_p)

    # Per-attack detection
    res = {'eps_learned': eps_l}
    for aname, ainfo in atk.items():
        Xa_t = (np.nan_to_num(ainfo['windows']) - mu) @ V
        n_n = len(Xn_p); n_at = len(Xa_t)
        y_true = np.concatenate([np.zeros(n_n), np.ones(n_at)])
        X_all = np.vstack([Xn_p, Xa_t])

        # Q-Direct per-window detection
        sq = per_window_scores(K_q, kmaker_q, Xn_p, X_all)
        # Learned surrogate per-window detection
        sl = per_window_scores(K_learned, kmaker_ls, Xn_p, X_all)

        res[aname] = dict(
            det_q=detect_metrics(y_true, sq, tau_sw_q),
            det_learned=detect_metrics(y_true, sl, tau_sw_ls),
        )
    return res


# ── Canonical attack order for each benchmark ─────────────────────

ATTACK_ORDER = {
    'CNC':  ['fdi', 'command', 'ramp', 'composite'],
    'CSTR': ['cstr_fdi', 'cstr_thermal', 'cstr_concentration'],
    'Tank': ['tank_bias', 'tank_oscillation', 'tank_drift'],
}

ATTACK_LABELS = {
    'fdi': 'FDI', 'command': 'Command', 'ramp': 'Ramp', 'composite': 'Composite',
    'cstr_fdi': 'FDI', 'cstr_thermal': 'Thermal', 'cstr_concentration': 'Conc.',
    'tank_bias': 'Bias', 'tank_oscillation': 'Oscill.', 'tank_drift': 'Drift',
}


# ── Main ──────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("  LEARNED-SURROGATE PER-WINDOW DETECTION (Supplement Table E)")
    print("  5 seeds × 3 benchmarks — Q-Direct + Learned-S only")
    print("=" * 70)

    all_res = {b: [] for b in ['CNC', 'CSTR', 'Tank']}

    for si, seed in enumerate(cfg.SEEDS):
        print(f"\n  Seed {seed} ({si+1}/5):")
        bench_raw = generate_bench_data(seed)
        for bname in ['CNC', 'CSTR', 'Tank']:
            nom, atk = bench_raw[bname][0], bench_raw[bname][1]
            print(f"    {bname}...", end=" ", flush=True)
            r = run_seed_surrogate_detect(nom, atk, seed)
            all_res[bname].append(r)
            print(f"eps_l={r['eps_learned']:.3f}")

    # ── Aggregate ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS: per-window F1 (mean ± std, 5 seeds)")
    print("=" * 70)

    print(f"\n  {'Bench':<6} {'Attack':<12} {'Q-Direct':>14} {'Learned-S':>14}")
    print("  " + "-" * 50)

    latex_rows = []
    for bname in ['CNC', 'CSTR', 'Tank']:
        data = all_res[bname]
        attacks = ATTACK_ORDER[bname]
        for a in attacks:
            f1_q = [d[a]['det_q']['f1'] for d in data]
            f1_l = [d[a]['det_learned']['f1'] for d in data]
            mq = np.mean(f1_q); sq = np.std(f1_q)
            ml = np.mean(f1_l); sl = np.std(f1_l)

            label = ATTACK_LABELS[a]
            print(f"  {bname:<6} {label:<12} {mq:.2f}±{sq:.2f}       {ml:.2f}±{sl:.2f}")

            # LaTeX row
            latex_rows.append(
                f"{bname:<5} & {label:<10} & {mq:.2f}$\\pm${sq:.2f} & {ml:.2f}$\\pm${sl:.2f} \\\\"
            )

        # Add midrule between benchmarks
        if bname != 'Tank':
            latex_rows.append("\\midrule")

    # ── Print LaTeX table body ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  LATEX TABLE BODY (paste into supplement_corrected.tex)")
    print("  Replace the TBD rows between \\midrule and \\bottomrule")
    print("=" * 70)
    print()
    for row in latex_rows:
        print(row)

    # ── Also print the surrogate epsilon summary ──────────────────
    print("\n  Surrogate ε (learned MLP, mean±std over 5 seeds):")
    for bname in ['CNC', 'CSTR', 'Tank']:
        eps = [d['eps_learned'] for d in all_res[bname]]
        print(f"    {bname}: ε_surr = {np.mean(eps):.3f}±{np.std(eps):.3f}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
