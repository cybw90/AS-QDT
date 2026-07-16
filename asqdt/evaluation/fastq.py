"""
Fast batched statevector evaluation for the fidelity quantum kernel,
plus shared evaluation utilities used by the v2 revision experiments.

The batched statevector computation is mathematically identical to
QuantumCircuit.statevector() (verified by unit test in run_v2_new_experiments)
but vectorized over samples, which makes Gram computation, adaptive-attack
optimization, and the sensor study tractable.
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from .. import config as cfg


# ── Batched statevectors ──────────────────────────────────────────

def batch_statevectors(qc, X, theta):
    """Compute U_theta(xi)|0> for every row of X. Returns (N, 2^nq) complex.

    Replicates QuantumCircuit.statevector exactly:
      per layer l: for each qubit i, RY(theta[l,i,0]*xi_j) then RZ(theta[l,i,1]*xi_j)
      with j = (l*nq + i) mod d, followed by the fixed entangling layer.
    Qubit 0 is the most significant bit (matches _kron_at ordering).
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    nq, depth = qc.nq, qc.depth
    psi = np.zeros((N, qc.dim), dtype=complex)
    psi[:, 0] = 1.0

    for l in range(depth):
        for i in range(nq):
            j = ((l * nq) + i) % d
            ay = theta[l, i, 0] * X[:, j]   # (N,)
            az = theta[l, i, 1] * X[:, j]
            cy, sy = np.cos(ay / 2), np.sin(ay / 2)
            ez_m, ez_p = np.exp(-1j * az / 2), np.exp(1j * az / 2)
            # Combined single-qubit gate G = RZ(az) @ RY(ay), per sample:
            # RY = [[c,-s],[s,c]], RZ = diag(e-, e+)
            G = np.empty((N, 2, 2), dtype=complex)
            G[:, 0, 0] = ez_m * cy
            G[:, 0, 1] = -ez_m * sy
            G[:, 1, 0] = ez_p * sy
            G[:, 1, 1] = ez_p * cy
            # Apply on qubit axis i (qubit 0 = most significant)
            psi = psi.reshape((N,) + (2,) * nq)
            psi = np.moveaxis(psi, 1 + i, 1)          # (N, 2, ...)
            shp = psi.shape
            psi = psi.reshape(N, 2, -1)
            psi = np.einsum('nab,nbk->nak', G, psi)
            psi = psi.reshape(shp)
            psi = np.moveaxis(psi, 1, 1 + i)
            psi = psi.reshape(N, qc.dim)
        psi = psi @ qc._ent.T
    return psi


def gram_from_sv(S1, S2=None):
    """Fidelity Gram |<psi_i|psi_j>|^2 from statevector batches."""
    if S2 is None:
        S2 = S1
    ov = S1.conj() @ S2.T
    return np.abs(ov) ** 2


class QKernelCache:
    """Caches statevectors for a fixed theta; fast gram/cross-gram."""

    def __init__(self, qc, theta):
        self.qc, self.theta = qc, theta

    def sv(self, X):
        return batch_statevectors(self.qc, X, self.theta)

    def gram(self, X, S=None):
        S = self.sv(X) if S is None else S
        K = gram_from_sv(S)
        np.fill_diagonal(K, 1.0)
        return K

    def cross(self, X, Y, SX=None, SY=None):
        SX = self.sv(X) if SX is None else SX
        SY = self.sv(Y) if SY is None else SY
        return gram_from_sv(SX, SY)


# ── MMD / per-window utilities (identical math to run_tifs_v2) ───

def mmd_u(K_XX, K_YY, K_XY):
    m, n = K_XX.shape[0], K_YY.shape[0]
    t1 = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
    t2 = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
    return t1 + t2 - 2.0 * np.sum(K_XY) / (m * n)


def pw_scores_from_grams(K_ref, K_tt_diag, K_rt):
    """S(xi_t) = meanoffdiag(K_ref) + k(t,t) - 2*mean_i k(ref_i, t).

    K_tt_diag: (T,) self-similarities; K_rt: (m, T) cross gram.
    Identical to per_window_scores() in run_tifs_v2.py, vectorized.
    """
    m = len(K_ref)
    mr = (np.sum(K_ref) - np.trace(K_ref)) / (m * (m - 1))
    return mr + K_tt_diag - 2.0 * K_rt.mean(axis=0)


def detect_f1(y_true, scores, thresh):
    return f1_score(y_true, (scores > thresh).astype(int), zero_division=0)


def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores)
    except Exception:
        return 0.5


def nominal_quantile(nom_scores, alpha=0.05):
    return np.percentile(nom_scores, 100 * (1 - alpha))


def worst_window_gap(nom_scores, atk_scores, alpha=0.05):
    """Standardized worst-window gap Delta_min (Supp. L, step 3)."""
    q = np.percentile(nom_scores, 100 * (1 - alpha))
    iqr = np.percentile(nom_scores, 75) - np.percentile(nom_scores, 25)
    return (np.min(atk_scores) - q) / (iqr + 1e-12)


def linear_cka(K1, K2):
    """Centered kernel alignment between two Gram matrices."""
    n = K1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K1c, K2c = H @ K1 @ H, H @ K2 @ H
    num = np.sum(K1c * K2c)
    den = np.sqrt(np.sum(K1c * K1c) * np.sum(K2c * K2c)) + 1e-12
    return num / den


def pca_fit(X, dim):
    X = np.nan_to_num(X)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
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


def design_quantum_kernel(qc, Xn_p, Xa_pool, seed, n_iters=cfg.MAX_ITERS, lr=cfg.KERNEL_LR):
    """Replicates run_tifs_v2.run_seed kernel optimization (SPSA, best-iterate),
    using the fast batched evaluator. Returns (theta_best, margins)."""
    rng = np.random.RandomState(seed)
    theta = qc.initial_params(rng)

    def obj(th):
        SX = batch_statevectors(qc, Xn_p, th)
        SY = batch_statevectors(qc, Xa_pool, th)
        Kxx = gram_from_sv(SX); np.fill_diagonal(Kxx, 1.0)
        Kyy = gram_from_sv(SY); np.fill_diagonal(Kyy, 1.0)
        return mmd_u(Kxx, Kyy, gram_from_sv(SX, SY))

    best_m, best_t = -np.inf, theta.copy()
    margins = []
    eps = 0.1
    for _ in range(n_iters):
        m_val = obj(theta)
        margins.append(m_val)
        if m_val > best_m:
            best_m, best_t = m_val, theta.copy()
        delta = rng.choice([-1, 1], size=theta.shape)
        g = (obj(theta + eps * delta) - obj(theta - eps * delta)) / (2 * eps) * delta
        theta += lr * (g - 0.02 * theta)
    return best_t, margins


def rbf_bandwidth(Xn_p, Xa_tune):
    """Replicates run_tifs_v2 bandwidth selection (median heuristic grid)."""
    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
    med = np.median(np.linalg.norm(Xn_p[:, None] - Xn_p[None, :], axis=2))
    gammas = [g / (med ** 2 + 1e-8) for g in [0.1, 0.5, 1, 2, 5]]
    best_g, best_v = gammas[2], -np.inf
    for g in gammas:
        v = mmd_u(sk_rbf(Xn_p, Xn_p, g), sk_rbf(Xa_tune, Xa_tune, g),
                  sk_rbf(Xn_p, Xa_tune, g))
        if v > best_v:
            best_v, best_g = v, g
    return best_g
