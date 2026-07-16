#!/usr/bin/env python3
"""
verify_sign_determinism.py — run AFTER applying sign_fix.patch.

Three checks (~60 s total, single core):

  1. CANONICALITY: the svd_flip-style convention maps 20 simulated
     backend sign patterns to one identical PCA basis.
  2. INVARIANCE: the quantum-kernel MMD at fixed theta is bit-identical
     under a simulated backend sign flip.
  3. CROSS-MACHINE REPRODUCTION: the full 5-seed CNC quantum design is
     re-run under the patched pipeline and compared (3 decimals) to the
     reference values produced on an independent machine
     (NumPy 2.4.4 / SciPy 1.17.1 / scikit-learn 1.8.0). Agreement here
     demonstrates that the quantum results are now environment-stable —
     the property the paper's reproducibility claim requires.

Usage:  python -m asqdt.scripts.verify_sign_determinism   (from repo root)
"""
import sys, os, warnings, time
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asqdt import config as cfg
from asqdt.cpms.data import generate_dataset
from asqdt.quantum.kernel import QuantumCircuit
from asqdt.detection.mmd import mmd_squared
from asqdt.scripts.run_tifs_v2 import pca_fit

# Reference values: patched pipeline, independent machine (2026-07-09),
# NumPy 2.4.4 / SciPy 1.17.1 / scikit-learn 1.8.0, x86-64 OpenBLAS.
REFERENCE_CNC_Q = {
    'fdi':       (+0.0132, 0.0119),
    'command':   (-0.0077, 0.0142),
    'ramp':      (+0.0115, 0.0128),
    'composite': (+0.0120, 0.0096),
}
REFERENCE_PER_SEED = {
    42:   {'fdi': -0.0097, 'command': -0.0152, 'ramp': -0.0001, 'composite': +0.0258},
    123:  {'fdi': +0.0132, 'command': +0.0139, 'ramp': +0.0208, 'composite': -0.0026},
    256:  {'fdi': +0.0192, 'command': -0.0164, 'ramp': +0.0297, 'composite': +0.0068},
    789:  {'fdi': +0.0241, 'command': -0.0243, 'ramp': -0.0049, 'composite': +0.0124},
    1024: {'fdi': +0.0189, 'command': +0.0038, 'ramp': +0.0121, 'composite': +0.0178},
}
TOL = 5e-4   # agreement at 3 displayed decimals


def _grams(qc, X, Y, theta):
    svX = [qc.statevector(x, theta) for x in X]
    svY = svX if Y is X else [qc.statevector(y, theta) for y in Y]
    KXX = np.eye(len(X)); KYY = np.eye(len(Y)); KXY = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            KXX[i, j] = KXX[j, i] = float(np.abs(np.vdot(svX[i], svX[j])) ** 2)
    for i in range(len(Y)):
        for j in range(i + 1, len(Y)):
            KYY[i, j] = KYY[j, i] = float(np.abs(np.vdot(svY[i], svY[j])) ** 2)
    for i in range(len(X)):
        for j in range(len(Y)):
            KXY[i, j] = float(np.abs(np.vdot(svX[i], svY[j])) ** 2)
    return KXX, KYY, KXY


def check_1_and_2():
    Xn, atk, *_ = generate_dataset(mismatch_frac=0.05, seed=42, W_L=10,
                                   E_max=2.0, legacy_method='chi_sq')
    mu, V = pca_fit(Xn, 12)
    real_svd = np.linalg.svd
    rng = np.random.RandomState(0)
    ok1 = True
    flips = None
    for _ in range(20):
        flips = rng.choice([-1.0, 1.0], size=12)
        def flipped_svd(A, full_matrices=False, _f=flips):
            U, S, Vt = real_svd(A, full_matrices=full_matrices)
            Vt = Vt.copy(); Vt[:12] *= _f[:, None]
            U = U.copy();  U[:, :12] *= _f[None, :]
            return U, S, Vt
        np.linalg.svd = flipped_svd
        try:
            mu2, V2 = pca_fit(Xn, 12)
        finally:
            np.linalg.svd = real_svd
        ok1 &= bool(np.allclose(V, V2) and np.allclose(mu, mu2))
    print(f"[1] Canonical PCA basis under 20 simulated backends: {'PASS' if ok1 else 'FAIL'}")

    qc = QuantumCircuit(4, 3)
    theta = qc.initial_params(np.random.RandomState(42))
    Xa = np.nan_to_num(atk['fdi']['windows'])

    def qmmd(Vm):
        Xn_p = (Xn - mu) @ Vm; Xa_p = (Xa - mu) @ Vm
        a, b, c = _grams(qc, Xn_p, Xa_p, theta)
        return mmd_squared(a, b, c)

    np.linalg.svd = flipped_svd
    try:
        _, V3 = pca_fit(Xn, 12)
    finally:
        np.linalg.svd = real_svd
    ok2 = abs(qmmd(V) - qmmd(V3)) < 1e-15
    print(f"[2] Quantum MMD backend-invariance at fixed theta:    {'PASS' if ok2 else 'FAIL'}")
    return ok1 and ok2


def _run_seed_q(seed, nq=4, depth=3):
    Xn, atk, *_ = generate_dataset(mismatch_frac=0.05, seed=seed, W_L=10,
                                   E_max=2.0, legacy_method='chi_sq')
    rng = np.random.RandomState(seed)
    mu, V = pca_fit(Xn, nq * depth)
    Xn_p = (Xn - mu) @ V
    qc = QuantumCircuit(nq, depth)
    theta = qc.initial_params(rng)
    best_m, best_t = -np.inf, theta.copy()
    Xa_pool = (np.nan_to_num(np.vstack([d['windows'] for d in atk.values()])) - mu) @ V

    def obj(th):
        a, b, c = _grams(qc, Xn_p, Xa_pool, th)
        return mmd_squared(a, b, c)

    for _ in range(cfg.MAX_ITERS):
        m_val = obj(theta)
        if m_val > best_m:
            best_m, best_t = m_val, theta.copy()
        delta = rng.choice([-1, 1], size=theta.shape)
        g = (obj(theta + 0.1 * delta) - obj(theta - 0.1 * delta)) / 0.2 * delta
        theta = theta + cfg.KERNEL_LR * (g - 0.02 * theta)
    theta = best_t
    res = {}
    for name, d in atk.items():
        Xa_t = (np.nan_to_num(d['windows']) - mu) @ V
        a, b, c = _grams(qc, Xn_p, Xa_t, theta)
        res[name] = mmd_squared(a, b, c)
    return res


def check_3():
    t0 = time.time()
    allres = {s: _run_seed_q(s) for s in cfg.SEEDS}
    ok = True
    print(f"[3] Cross-machine reproduction ({time.time()-t0:.0f}s):")
    print(f"    {'family':10s} {'this machine':>16s} {'reference':>16s}")
    for fam, (rmean, rstd) in REFERENCE_CNC_Q.items():
        vals = np.array([allres[s][fam] for s in cfg.SEEDS])
        hit = abs(vals.mean() - rmean) < TOL and abs(vals.std() - rstd) < TOL
        ok &= hit
        print(f"    {fam:10s} {vals.mean():+.4f}±{vals.std():.4f} "
              f"  {rmean:+.4f}±{rstd:.4f}  {'PASS' if hit else 'FAIL'}")
    for s in cfg.SEEDS:
        for fam, rv in REFERENCE_PER_SEED[s].items():
            if abs(allres[s][fam] - rv) >= TOL:
                ok = False
                print(f"    seed {s} {fam}: {allres[s][fam]:+.4f} vs ref {rv:+.4f}  FAIL")
    print(f"    per-seed values: {'PASS' if ok else 'see failures above'}")
    return ok


if __name__ == '__main__':
    ok = check_1_and_2()
    ok &= check_3()
    print("\nOVERALL:", "PASS — quantum pipeline is environment-stable; "
          "proceed with the full re-run." if ok else
          "FAIL — do not re-run tables; report the failing values.")
    sys.exit(0 if ok else 1)
