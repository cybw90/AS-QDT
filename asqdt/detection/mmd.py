"""
MMD-based detection: empirical MMD² statistic, threshold calibration, detection test.

Implements Eq. (9)-(11) and Theorem 1 / Corollary 1 from the paper.
"""
import numpy as np
from .. import config as cfg


def mmd_squared(K_XX, K_YY, K_XY):
    """
    Unbiased MMD² U-statistic from precomputed kernel matrices.
    
    MMD²_u = 1/(m(m-1)) Σ_{i≠i'} K_XX[i,i']
           + 1/(n(n-1)) Σ_{j≠j'} K_YY[j,j']
           - 2/(mn) Σ_{i,j} K_XY[i,j]
    """
    m = K_XX.shape[0]
    n = K_YY.shape[0]
    
    # Zero diagonals for unbiased estimator
    term1 = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
    term2 = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
    term3 = 2.0 * np.sum(K_XY) / (m * n)
    
    return term1 + term2 - term3


def mmd_from_embeddings(phi_X, phi_Y):
    """MMD² from surrogate embeddings φ̃(X) and φ̃(Y)."""
    K_XX = phi_X @ phi_X.T
    K_YY = phi_Y @ phi_Y.T
    K_XY = phi_X @ phi_Y.T
    return mmd_squared(K_XX, K_YY, K_XY)


def compute_threshold(kappa, m, n, delta, eps_surr):
    """
    Threshold τ_δ = β_{m,n,δ} + 4·ε_surr  (Eq. 12).
    
    β = 4κ²·√(½·log(2/δ)·(1/m + 1/n))
    Direct McDiarmid on unbiased U-statistic (Theorem 1).
    """
    beta = 4.0 * kappa**2 * np.sqrt(0.5 * np.log(2.0/delta) * (1.0/m + 1.0/n))
    tau = beta + 4 * eps_surr
    return tau, beta


def permutation_threshold(phi_X_nom, alpha=cfg.DELTA, n_perm=500):
    """
    Calibrate threshold from nominal data via permutation test.
    More practical than the analytic bound for finite samples.
    """
    N = len(phi_X_nom)
    m = N // 2
    n = N - m
    null_stats = []
    rng = np.random.RandomState(0)
    for _ in range(n_perm):
        perm = rng.permutation(N)
        pX = phi_X_nom[perm[:m]]
        pY = phi_X_nom[perm[m:m+n]]
        null_stats.append(mmd_from_embeddings(pX, pY))
    null_stats = np.sort(null_stats)
    idx = int(np.ceil((1 - alpha) * n_perm)) - 1
    return null_stats[min(idx, len(null_stats) - 1)]


class MMDDetector:
    """Online MMD detector using surrogate embeddings."""

    def __init__(self, phi_nom, tau_delta, tau_perm=None):
        """
        Args:
            phi_nom: (m, D) surrogate embeddings of nominal reference windows
            tau_delta: analytic threshold from Theorem 1
            tau_perm: permutation-calibrated threshold (optional, preferred)
        """
        self.phi_nom = phi_nom
        self.tau = tau_perm if tau_perm is not None else tau_delta
        self.tau_analytic = tau_delta
        self.tau_perm = tau_perm

    def test(self, phi_online):
        """
        Run MMD test on online surrogate embeddings.
        
        Returns: (alarm: bool, T_t: float, tau: float)
        """
        T_t = mmd_from_embeddings(self.phi_nom, phi_online)
        return T_t > self.tau, T_t, self.tau

    def margin_estimate(self, phi_attack):
        """Estimate Γ̂ = MMD²(nominal, attack) at current attack iterate."""
        return mmd_from_embeddings(self.phi_nom, phi_attack)
